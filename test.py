import argparse
import numpy as np
import tensorflow as tf

np.random.seed(1234)

def main(args):
    # set your data
    # |test_x| = (n, 8, 5000) => (number of data, 8-lead, 5000 data samples)
    # |test_y| = (n, 2) => (number of data, [1, 0] or [0, 1])
    test_x = np.load(args.test_x_path, allow_pickle=True) 
    test_y = np.load(args.test_y_path, allow_pickle=True)

    # correct parameter to your data
    test_x = normalize_test(test_x, 
                            scaler_path
                            num_leads=12, 
                            seq_len=5000)

    test_p_waves = your_make_pwave_function() # make your p-waves
    
    test_x = np.swapaxes(test_x, 1, 2) # |test_x| = (n, 4000, 12)

    model = tf.keras.models.load_model(args.model_path)
    

    # inference results
    preds = model.predict([test_x, test_p_waves], verbose=1)

# ArgParser 설정
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference the model")
    
    parser.add_argument('--num_classes', type=int, default=2, help="Number of classes (default: 2)")

    # model hyperparameters
    parser.add_argument('--test_x_path', type=str, default='./data/test_x.npy', help="test_x numpy file path (default: ./data/test_x.npy)")
    parser.add_argument('--test_y_path', type=str, default='./data/test_y.npy', help="test_y numpy file path (default: ./data/test_y.npy)")
    parser.add_argument('--scaler_path', type=str, default='./save_models/paf_scaler.pkl', help="saved scaler path (default: ./save_models/paf_scaler.pkl)")
    parser.add_argument('--model_path', type=str, default='./save_models/paf.h5', help="saved model path (default: ./save_models/paf.h5')")
    
    
    args = parser.parse_args()
    main(args)