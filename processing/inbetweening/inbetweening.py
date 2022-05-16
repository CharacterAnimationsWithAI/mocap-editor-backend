import os
import torch
import numpy as np
from tqdm import tqdm
from .models import StateEncoder, OffsetEncoder, TargetEncoder, LSTM, Decoder
from .skeleton.skeleton import Skeleton
from .functions import gen_ztta, write_to_bvhfile
from .lafan1.config import *
from .lafan1.dataset import LaFan1

class Inbetweening:

    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.load_models()
        self.load_skeleton()


    def load_models(self):
        models_dir = "./processing/inbetweening/models"

        self.state_encoder = StateEncoder(in_dim=model["state_input_dim"])
        self.state_encoder = self.state_encoder.to(self.device)
        self.state_encoder.load_state_dict(
            torch.load(f"{models_dir}/state_encoder.pkl", 
            map_location=torch.device(self.device)))

        self.offset_encoder = OffsetEncoder(in_dim=model["offset_input_dim"])
        self.offset_encoder = self.offset_encoder.to(self.device)
        self.offset_encoder.load_state_dict(
            torch.load(f"{models_dir}/offset_encoder.pkl", 
            map_location=torch.device(self.device)))

        self.target_encoder = TargetEncoder(in_dim=model["target_input_dim"])
        self.target_encoder = self.target_encoder.to(self.device)
        self.target_encoder.load_state_dict(
            torch.load(f"{models_dir}/target_encoder.pkl",
            map_location=torch.device(self.device)))

        self.lstm = LSTM(in_dim=model["lstm_dim"], hidden_dim=model["lstm_dim"] * 2)
        self.lstm = self.lstm.to(self.device)
        self.lstm.load_state_dict(
            torch.load(f"{models_dir}/lstm.pkl", 
            map_location=torch.device(self.device)))

        self.decoder = Decoder(in_dim=model["lstm_dim"] * 2, out_dim=model["decoder_output_dim"])
        self.decoder = self.decoder.to(self.device)
        self.decoder.load_state_dict(
            torch.load(f"{models_dir}/decoder.pkl",
            map_location=torch.device(self.device)))

        self.state_encoder.eval()
        self.offset_encoder.eval()
        self.target_encoder.eval()
        self.lstm.eval()
        self.decoder.eval()


    def load_skeleton(self):
        skeleton = Skeleton(offsets=data["offsets"], parents=data["parents"])
        skeleton.to(self.device)
        skeleton.remove_joints(data["joints_to_remove"])

    
    def inbetween(self, input_bvh_path, num_seed_frames):
        sequence = LaFan1.load_single_bvh_sequence(input_bvh_path)
        sequence_length = sequence['X'].shape[0]

        with torch.no_grad():
            # State inputs
            local_q = torch.tensor(sequence['local_q'], dtype=torch.float32).unsqueeze(0).to(self.device)
            root_v = torch.tensor(sequence['root_v'], dtype=torch.float32).unsqueeze(0).to(self.device)
            contact = torch.tensor(sequence['contact'], dtype=torch.float32).unsqueeze(0).to(self.device)

            # Offset inputs
            root_p_offset = torch.tensor(sequence['root_p_offset'], dtype=torch.float32).unsqueeze(0).to(self.device)
            local_q_offset = torch.tensor(sequence['local_q_offset'], dtype=torch.float32).unsqueeze(0).to(self.device)
            local_q_offset = local_q_offset.view(local_q_offset.size(0), -1)

            # Target inputs
            target = torch.tensor(sequence['target'], dtype=torch.float32).unsqueeze(0).to(self.device)
            target = target.view(target.size(0), -1)

            # Root position
            root_p = torch.tensor(sequence['root_p'], dtype=torch.float32).unsqueeze(0).to(self.device)

            # X
            X = torch.tensor(sequence['X'], dtype=torch.float32).unsqueeze(0).to(self.device)

            self.lstm.init_hidden(local_q.size(0))

            root_pred = None
            local_q_pred = None
            contact_pred = None
            root_v_pred = None
            bvh_list = []
            bvh_list.append(torch.cat([X[:, 0, 0], local_q[:, 0,].view(local_q.size(0), -1)], -1))

            