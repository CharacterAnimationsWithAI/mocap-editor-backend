import os
import torch
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

    
    def inbetween(self, input_bvh_path, num_seed_frames, output_path):
        sequence = LaFan1.load_single_bvh_sequence(input_bvh_path)
        sequence_length = sequence['X'].shape[0]
        ztta = gen_ztta(timesteps=sequence_length).to(self.device)

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

            for t in tqdm(range(sequence_length - 1)):
                if t  == 0:
                    root_p_t = root_p[:,t]
                    local_q_t = local_q[:,t]
                    local_q_t = local_q_t.view(local_q_t.size(0), -1)
                    contact_t = contact[:,t]
                    root_v_t = root_v[:,t]
                else:
                    root_p_t = root_pred[0]
                    local_q_t = local_q_pred[0]
                    contact_t = contact_pred[0]
                    root_v_t = root_v_pred[0]

                state_input = torch.cat([local_q_t, root_v_t, contact_t], -1)

                root_p_offset_t = root_p_offset - root_p_t
                local_q_offset_t = local_q_offset - local_q_t
                offset_input = torch.cat([root_p_offset_t, local_q_offset_t], -1)

                target_input = target

                h_state = self.state_encoder(state_input)
                h_offset = self.offset_encoder(offset_input)
                h_target = self.target_encoder(target_input)

                tta = sequence_length - t - 2
                
                h_state += ztta[tta]
                h_offset += ztta[tta]
                h_target += ztta[tta]
                
                if tta < 5:
                    lambda_target = 0.0
                elif tta >= 5 and tta < 30:
                    lambda_target = (tta - 5) / 25.0
                else:
                    lambda_target = 1.0
                h_offset += 0.5 * lambda_target * torch.FloatTensor(h_offset.size()).normal_().to(self.device)
                h_target += 0.5 * lambda_target * torch.FloatTensor(h_target.size()).normal_().to(self.device)

                h_in = torch.cat([h_state, h_offset, h_target], -1).unsqueeze(0)
                h_out = self.lstm(h_in)

                h_pred, contact_pred = self.decoder(h_out)
                local_q_v_pred = h_pred[:, :, :88]
                local_q_pred = local_q_v_pred + local_q_t

                local_q_pred_ = local_q_pred.view(local_q_pred.size(0), local_q_pred.size(1), -1, 4)
                local_q_pred_ = local_q_pred_ / torch.norm(local_q_pred_, dim = -1, keepdim = True)

                root_v_pred = h_pred[:,:,88:]
                root_pred = root_v_pred + root_p_t

                bvh_list.append(torch.cat([root_pred[0], local_q_pred_[0].view(local_q_pred.size(1), -1)], -1))

                local_q_next = local_q[:,t+1]
                local_q_next = local_q_next.view(local_q_next.size(0), -1)

            bvh_data = torch.cat([x[0].unsqueeze(0) for x in bvh_list], 0).detach().cpu().numpy()
            write_to_bvhfile(bvh_data, output_path, data['joints_to_remove'])
                