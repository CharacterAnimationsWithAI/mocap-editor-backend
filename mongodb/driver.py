from pymongo import MongoClient

class Driver:
    def __init__(self, url):
        self.client = MongoClient(url)
        self.database = self.client["mocapEditorDB"]
        self.check_initial_statistics()
        print("Database connection established...")

    def insert_log(self, data_dict):
        logs_collection = self.database["logs"]
        logs_collection.insert_one(data_dict)

    def get_logs(self):
        response = []
        logs_collection = self.database["logs"]
        results = logs_collection.find({}, {'_id': False}).sort("date", -1).limit(7)

        for result in results.clone():
            result["date"] = str(result["date"])
            response += [result]

        return response

    def check_initial_statistics(self):
        statistics_collection = self.database["statistics"]

        if not list(statistics_collection.find({})):
            statistics_collection.insert_many([{"average_motion_inference_time": 0.0}, {"average_style_transfer_time": 0.0}, {"average_bvh_length": 0.0}])

    def update_average_motion_inference_time(self, timedelta):
        statistics_collection = self.database["statistics"]
        current_data = statistics_collection.find({}).clone()
        current_time = list(current_data)[0]["average_motion_inference_time"]

        statistics_collection.update_one({"average_motion_inference_time": current_time}, {"$set": {"average_motion_inference_time": current_time + timedelta}})

    def update_average_style_transfer_time(self, timedelta):
        statistics_collection = self.database["statistics"]
        current_data = statistics_collection.find({}).clone()
        current_time = list(current_data)[1]["average_style_transfer_time"]

        statistics_collection.update_one({"average_style_transfer_time": current_time}, {"$set": {"average_style_transfer_time": current_time + timedelta}})

    def update_average_bvh_length(self, timedelta):
        statistics_collection = self.database["statistics"]
        current_data = statistics_collection.find({}).clone()
        current_time = list(current_data)[2]["average_bvh_length"]

        statistics_collection.update_one({"average_bvh_length": current_time}, {"$set": {"average_bvh_length": current_time + timedelta}})
    
    def get_average_motion_inference_time(self):
        statistics_collection = self.database["statistics"]
        current_data = statistics_collection.find({}).clone()
        current_time = list(current_data)[0]["average_motion_inference_time"]
        
        return current_time

    def get_average_style_transfer_time(self):
        statistics_collection = self.database["statistics"]
        current_data = statistics_collection.find({}).clone()
        current_time = list(current_data)[1]["average_style_transfer_time"]
        
        return current_time
    
    def get_average_bvh_length(self):
        statistics_collection = self.database["statistics"]
        current_data = statistics_collection.find({}).clone()
        current_time = list(current_data)[2]["average_bvh_length"]

        return current_time