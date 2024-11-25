import argparse
import numpy as np
from models.model import BuildModel

np.random.seed(1234)

def main(args):
    # set your data
    # |train_x| = (n, 8, 5000) => (number of data, 8-lead, 5000 data samples)
    # |train_y| = (n, 2) => (number of data, [1, 0] or [0, 1])
    train_x = np.load('./train_x.npy', allow_pickle=True) 
    train_y = np.load('./train_y.npy', allow_pickle=True)

    valid_x = np.load('./valid_x.npy', allow_pickle=True) 
    valid_y = np.load('./valid_y.npy', allow_pickle=True)

    # correct parameter to your data
    train_x, valid_x = normalize(train_x, valid_x, 
                                  scaler=None, scaler_path='./save_models/paf_scaler.pkl',
                                  num_leads=12, seq_len=5000)

    train_p_waves = your_make_pwave_function() # make your p-waves
    valid_p_waves = your_make_pwave_function() # make your p-waves
    
    train_x = np.swapaxes(train_x, 1, 2) # |train_x| = (n, 4000, 12)
    valid_x = np.swapaxes(valid_x, 1, 2)

    # 모델 하이퍼파라미터 설정
    kargs = {
        'conv1_filter': args.conv1_filter,
        'conv2_filter': args.conv2_filter,
        'conv3_filter': args.conv3_filter,
        'conv1x1_filter': args.conv1x1_filter,
        'conv1_kernel_size': args.conv1_kernel_size,
        'conv2_kernel_size': args.conv2_kernel_size,
        'conv3_kernel_size': args.conv3_kernel_size,
        'conv1x1_kernel_size': args.conv1x1_kernel_size,
        'dilation_rate': args.dilation_rate,
        'lstm1_units': args.lstm1_units,
        'lstm2_units': args.lstm2_units,
        'lstm3_units': args.lstm3_units,
        'lstm4_units': args.lstm4_units,
        'fc1': args.fc1,
        'fc2': args.fc2,
        'num_classes': args.num_classes
    }

    # 모델 생성
    model = BuildModel(**kargs)

    # 모델 컴파일
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # 모델 학습
    model.fit([train_x, train_p_waves], 
              train_y, 
              epochs=args.epochs, 
              batch_size=args.batch_size,
              valid_data=([valid_x, valid_p_waves], valid_y),
              verbose=1)

    # set your save path
    model.save('./save_models/paf.h5')

# ArgParser 설정
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model")
    
    parser.add_argument('--num_classes', type=int, default=2, help="Number of classes (default: 2)")

    # model hyperparameters
    parser.add_argument('--conv1_filter', type=int, default=16, help="Conv0 filter size (default: 16)")
    parser.add_argument('--conv2_filter', type=int, default=32, help="Conv1 filter size (default: 32)")
    parser.add_argument('--conv3_filter', type=int, default=64, help="Conv2 filter size (default: 64)")
    parser.add_argument('--conv1x1_filter', type=int, default=128, help="Conv1x1 filter size (default: 128)")
    parser.add_argument('--conv1_kernel_size', type=int, default=3, help="Conv0 kernel size (default: 3)")
    parser.add_argument('--conv2_kernel_size', type=int, default=3, help="Conv1 kernel size (default: 3)")
    parser.add_argument('--conv3_kernel_size', type=int, default=3, help="Conv2 kernel size (default: 3)")
    parser.add_argument('--conv1x1_kernel_size', type=int, default=1, help="Conv1x1 kernel size (default: 1)")
    parser.add_argument('--dilation_rate', type=int, default=2, help="Dilation rate (default: 2)")

    parser.add_argument('--lstm1_units', type=int, default=8, help="LSTM1 units (default: 8)")
    parser.add_argument('--lstm2_units', type=int, default=16, help="LSTM2 units (default: 16)")
    parser.add_argument('--lstm3_units', type=int, default=32, help="LSTM3 units (default: 32)")
    parser.add_argument('--lstm4_units', type=int, default=64, help="LSTM4 units (default: 64)")

    parser.add_argument('--fc1', type=int, default=64, help="Fully connected layer 1 size (default: 128)")
    parser.add_argument('--fc2', type=int, default=32, help="Fully connected layer 2 size (default: 64)")

    parser.add_argument('--epochs', type=int, default=100, help="Number of epochs (default: 100)")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size (default: 32)")
    
    args = parser.parse_args()
    main(args)