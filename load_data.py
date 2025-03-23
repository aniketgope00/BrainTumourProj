import kagglehub

def load_tumor_data()->None:
    # Download latest version
    path = kagglehub.dataset_download("navoneel/brain-mri-images-for-brain-tumor-detection")

    print("Path to dataset files:", path)

if __name__ == "__main__":
    load_tumor_data()
    #TODO
    #move_dir() - move the downloaded dataset to the desired location