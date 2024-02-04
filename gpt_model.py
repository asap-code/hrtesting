from skllm.config import SKLLMConfig
from skllm.models.gpt.classification.few_shot import FewShotGPTClassifier
import pandas as pd
from sklearn.model_selection import train_test_split

class GPTModel:
    def __init__(self):
        # Set up the configuration
        SKLLMConfig.set_gpt_key("4b81012d55fb416c9e398f6149c3071e")
        SKLLMConfig.set_azure_api_base("https://ey-sandbox.openai.azure.com/")

        # Initialize the classifier
        self.clf = FewShotGPTClassifier(model="azure::gpt-4-32k")

    def load_and_process_data(self, excel_file_path, test_size=0.2, random_seed=None):
        df = pd.read_excel(excel_file_path)
        # Processing logic here...
        team_records = {
        'Hello HR': df[df['Team'] == 'Hello HR'].sample(frac=1, random_state=random_seed),
        'Fleet': df[df['Team'] == 'Fleet'].sample(frac=1, random_state=random_seed),
        'BB/BI/VIP': df[df['Team'] == 'BB/BI/VIP'].sample(frac=1, random_state=random_seed),
        'Shared Process': df[df['Team'] == 'Shared Process'].sample(frac=1, random_state=random_seed)
        }

        # Combining all teams' records and splitting into training and validation sets
        all_records = pd.concat(team_records.values())
        training_set, validation_set = train_test_split(all_records, test_size=test_size, random_state=random_seed)

        return training_set['Content mail'].to_list(), training_set['Team'].to_list(), validation_set['Content mail'].to_list(), validation_set['Team'].to_list()
        

    def train(self, training_text, training_labels):
        
        self.clf.fit(training_text, training_labels)

    def predict(self, input_data):
        return self.clf.predict([input_data])[0]  # Assuming prediction for single instance