
# Define constants for the dataset and output paths
api = HfApi(token=os.getenv("HF_TOKEN"))
DATASET_PATH = "hf://datasets/mkrish2025/Machine-Failure-Prediction/data/engine_data.csv"
df = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully.")

repo_id = "mkrish2025/Machine-Failure-Prediction"
repo_type = "dataset"

#Applying preprocessing

df.columns = df.columns.str.replace(' ', '_').str.lower()
df = df.round(3)

target_col = 'engine_condition'

# Split into X (features) and y (target)
X = df.drop(columns=[target_col])
y = df[target_col]

# Perform train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

#local_dir = "mc_failure_prediction/data"

# X_train.to_csv(f"{local_dir}/Xtrain.csv", index=False)
# X_test.to_csv(f"{local_dir}/Xtest.csv", index=False)
# y_train.to_csv(f"{local_dir}/ytrain.csv", index=False)
# y_test.to_csv(f"{local_dir}/ytest.csv", index=False)

X_train.to_csv("Xtrain.csv", index=False)
X_test.to_csv("Xtest.csv", index=False)
y_train.to_csv("ytrain.csv", index=False)
y_test.to_csv("ytest.csv", index=False)


files = ["Xtrain.csv","Xtest.csv","ytrain.csv","ytest.csv"]

# Step 1: Check if the space exists
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Space '{repo_id}' already exists. Using it.")
except RepositoryNotFoundError:
    print(f"Space '{repo_id}' not found. Creating new space...")
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
    print(f"Space '{repo_id}' created.")

for file_name in files:
    api.upload_file(
        path_or_fileobj=file_name,
        path_in_repo=file_name.split("/")[-1],  # just the filename
        repo_id="mkrish2025/Machine-Failure-Prediction",
        repo_type="dataset",
    )
