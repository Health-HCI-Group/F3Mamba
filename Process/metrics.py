import numpy as np
import pandas as pd
import torch
from PostProcess import *
from tqdm import tqdm
from scipy.stats import gaussian_kde

def read_label(dataset):
    """Read manually corrected labels."""
    df = pd.read_csv("label/{0}_Comparison.csv".format(dataset))
    out_dict = df.to_dict(orient='index')
    out_dict = {str(value['VideoID']): value for key, value in out_dict.items()}
    return out_dict


def read_hr_label(feed_dict, index):
    """Read manually corrected UBFC labels."""
    # For UBFC only
    if index[:7] == 'subject':
        index = index[7:]
    video_dict = feed_dict[index]
    if video_dict['Preferred'] == 'Peak Detection':
        hr = video_dict['Peak Detection']
    elif video_dict['Preferred'] == 'FFT':
        hr = video_dict['FFT']
    else:
        hr = video_dict['Peak Detection']
    return index, hr


def _reform_data_from_dict(data, flatten=True):
    """Helper func for calculate metrics: reformat predictions and labels from dicts. """
    sort_data = sorted(data.items(), key=lambda x: x[0])
    sort_data = [i[1] for i in sort_data]
    sort_data = torch.cat(sort_data, dim=0)

    if flatten:
        sort_data = np.reshape(sort_data.cpu(), (-1))
    else:
        sort_data = np.array(sort_data.cpu())

    return sort_data


def calculate_metrics(predictions, labels, config, datatype):
    """Calculate rPPG Metrics (MAE, RMSE, MAPE, Pearson Coef.)."""
    predict_hr_fft_all = list()
    gt_hr_fft_all = list()
    predict_hr_peak_all = list()
    gt_hr_peak_all = list()
    SNR_all = list()
    MACC_all = list()

    TEST_DATA_FS  =30
    diff_flag_test = False
    EVALUATION_METHOD = "FFT"
    TEST_METRICS = []

    print(f"{datatype} ----------------- Calculating metrics!")
    for index in tqdm(predictions.keys(), ncols=80):
        prediction = _reform_data_from_dict(predictions[index])
        label = _reform_data_from_dict(labels[index])

        video_frame_size = prediction.shape[0]

        window_frame_size = video_frame_size

        for i in range(0, len(prediction), window_frame_size):
            pred_window = prediction[i:i+window_frame_size]
            label_window = label[i:i+window_frame_size]

            if len(pred_window) < 9:
                print(f"Window frame size of {len(pred_window)} is smaller than minimum pad length of 9. Window ignored!")
                continue
            
            if EVALUATION_METHOD == "peak detection":
                gt_hr_peak, pred_hr_peak, SNR, macc = calculate_metric_per_video(
                    pred_window, label_window, diff_flag=diff_flag_test, fs=TEST_DATA_FS, hr_method='Peak')
                gt_hr_peak_all.append(gt_hr_peak)
                predict_hr_peak_all.append(pred_hr_peak)
                if datatype == "rppg":
                    SNR_all.append(SNR)
                MACC_all.append(macc)
            elif EVALUATION_METHOD == "FFT":
                gt_hr_fft, pred_hr_fft, SNR, macc = calculate_metric_per_video(
                    pred_window, label_window, diff_flag=diff_flag_test, fs=TEST_DATA_FS, hr_method='FFT', datatype=datatype)
                gt_hr_fft_all.append(gt_hr_fft)
                predict_hr_fft_all.append(pred_hr_fft)
                if datatype == "rppg":
                    SNR_all.append(SNR)
                MACC_all.append(macc)
            else:
                raise ValueError("Inference evaluation method name wrong!")
    
    def to_numpy(x):
        if isinstance(x, torch.Tensor):
            return x.numpy()
        return x

    gt_hr_fft_all = np.array([to_numpy(tensor) for tensor in gt_hr_fft_all])
    predict_hr_fft_all = np.array([to_numpy(tensor) for tensor in predict_hr_fft_all])
    if datatype == "rppg":
        SNR_all = np.array([to_numpy(tensor) for tensor in SNR_all])
    MACC_all = np.array([to_numpy(tensor) for tensor in MACC_all])

    metrics_result = {}
    if EVALUATION_METHOD == "FFT":
        num_test_samples = len(predict_hr_fft_all)
        
        predict_hr_fft_all = [np.array(x, ndmin=1) for x in predict_hr_fft_all]
        predict_hr_fft_all = np.concatenate(predict_hr_fft_all)
        gt_hr_fft_all = [np.array(x, ndmin=1) for x in gt_hr_fft_all]
        gt_hr_fft_all = np.concatenate(gt_hr_fft_all)
    
        for metric in TEST_METRICS:
            if metric == "MAE":
                MAE_FFT = np.mean(np.abs(predict_hr_fft_all - gt_hr_fft_all))
                MAE_STD = np.std(np.abs(predict_hr_fft_all - gt_hr_fft_all)) / np.sqrt(num_test_samples)
                metrics_result["FFT_MAE"] = (MAE_FFT, MAE_STD)
                print("FFT MAE (FFT Label): {0} +/- {1}".format(MAE_FFT, MAE_STD))
            elif metric == "RMSE":
                RMSE_FFT = np.sqrt(np.mean(np.square(predict_hr_fft_all - gt_hr_fft_all)))
                RMSE_STD = np.std(np.square(predict_hr_fft_all - gt_hr_fft_all)) / np.sqrt(num_test_samples)
                metrics_result["FFT_RMSE"] = (RMSE_FFT, RMSE_STD)
                print("FFT RMSE (FFT Label): {0} +/- {1}".format(RMSE_FFT, RMSE_STD))
            elif metric == "MAPE":
                MAPE_FFT = np.mean(np.abs((predict_hr_fft_all - gt_hr_fft_all) / gt_hr_fft_all)) * 100
                MAPE_STD = np.std(np.abs((predict_hr_fft_all - gt_hr_fft_all) / gt_hr_fft_all)) / np.sqrt(num_test_samples) * 100
                metrics_result["FFT_MAPE"] = (MAPE_FFT, MAPE_STD)
                print("FFT MAPE (FFT Label): {0} +/- {1}".format(MAPE_FFT, MAPE_STD))
            elif metric == "Pearson":
                Pearson_FFT = np.corrcoef(predict_hr_fft_all, gt_hr_fft_all)
                correlation_coefficient = Pearson_FFT[0][1]
                Pearson_STD = np.sqrt((1 - correlation_coefficient**2) / (num_test_samples - 2))
                metrics_result["FFT_Pearson"] = (correlation_coefficient, Pearson_STD)
                print("FFT Pearson (FFT Label): {0} +/- {1}".format(correlation_coefficient, Pearson_STD))
            elif metric == "SNR" and datatype == "rppg":
                if len(SNR_all) > 0:
                    SNR_FFT = np.mean(SNR_all)
                    SNR_STD = np.std(SNR_all) / np.sqrt(num_test_samples)
                    metrics_result["FFT_SNR"] = (SNR_FFT, SNR_STD)
                    print("FFT SNR (FFT Label): {0} +/- {1} (dB)".format(SNR_FFT, SNR_STD))
                else:
                    print("FFT SNR (FFT Label): No valid SNR values")
            elif metric == "MACC":
                MACC_avg = np.mean(MACC_all)
                MACC_STD = np.std(MACC_all) / np.sqrt(num_test_samples)
                metrics_result["MACC"] = (MACC_avg, MACC_STD)
                print("MACC: {0} +/- {1}".format(MACC_avg, MACC_STD))
            else:
                if metric == "SNR":
                    print(f"Skipping SNR for datatype {datatype}.")
                    continue
                print(f"Unexpected metric type: {metric}")
                raise ValueError("Wrong Test Metric Type")
            
    elif EVALUATION_METHOD == "peak detection":
        gt_hr_peak_all = np.array([to_numpy(tensor) for tensor in gt_hr_peak_all])
        predict_hr_peak_all = np.array([to_numpy(tensor) for tensor in predict_hr_peak_all])
        if datatype == "rppg":
            SNR_all = np.array([to_numpy(tensor) for tensor in SNR_all])
        MACC_all = np.array([to_numpy(tensor) for tensor in MACC_all])
        
        num_test_samples = len(predict_hr_peak_all)
        for metric in TEST_METRICS:
            if metric == "MAE":
                MAE_PEAK = np.mean(np.abs(predict_hr_peak_all - gt_hr_peak_all))
                MAE_STD = np.std(np.abs(predict_hr_peak_all - gt_hr_peak_all)) / np.sqrt(num_test_samples)
                metrics_result["PEAK_MAE"] = (MAE_PEAK, MAE_STD)
                print("Peak MAE (Peak Label): {0} +/- {1}".format(MAE_PEAK, MAE_STD))
            elif metric == "RMSE":
                RMSE_PEAK = np.sqrt(np.mean(np.square(predict_hr_peak_all - gt_hr_peak_all)))
                RMSE_STD = np.std(np.square(predict_hr_peak_all - gt_hr_peak_all)) / np.sqrt(num_test_samples)
                metrics_result["PEAK_RMSE"] = (RMSE_PEAK, RMSE_STD)
                print("PEAK RMSE (Peak Label): {0} +/- {1}".format(RMSE_PEAK, RMSE_STD))
            elif metric == "MAPE":
                MAPE_PEAK = np.mean(np.abs((predict_hr_peak_all - gt_hr_peak_all) / gt_hr_peak_all)) * 100
                MAPE_STD = np.std(np.abs((predict_hr_peak_all - gt_hr_peak_all) / gt_hr_peak_all)) / np.sqrt(num_test_samples) * 100
                metrics_result["PEAK_MAPE"] = (MAPE_PEAK, MAPE_STD)
                print("PEAK MAPE (Peak Label): {0} +/- {1}".format(MAPE_PEAK, MAPE_STD))
            elif metric == "Pearson":
                Pearson_PEAK = np.corrcoef(predict_hr_peak_all, gt_hr_peak_all)
                correlation_coefficient = Pearson_PEAK[0][1]
                Pearson_STD = np.sqrt((1 - correlation_coefficient**2) / (num_test_samples - 2))
                metrics_result["PEAK_Pearson"] = (correlation_coefficient, Pearson_STD)
                print("PEAK Pearson (Peak Label): {0} +/- {1}".format(correlation_coefficient, Pearson_STD))
            elif metric == "SNR" and datatype == "rppg":
                if len(SNR_all) > 0:
                    SNR_PEAK = np.mean(SNR_all)
                    SNR_STD = np.std(SNR_all) / np.sqrt(num_test_samples)
                    metrics_result["PEAK_SNR"] = (SNR_PEAK, SNR_STD)
                    print("PEAK SNR (Peak Label): {0} +/- {1} (dB)".format(SNR_PEAK, SNR_STD))
                else:
                    print("PEAK SNR (Peak Label): No valid SNR values")
            elif metric == "MACC":
                MACC_avg = np.mean(MACC_all)
                MACC_STD = np.std(MACC_all) / np.sqrt(num_test_samples)
                metrics_result["MACC"] = (MACC_avg, MACC_STD)
                print("MACC: {0} +/- {1}".format(MACC_avg, MACC_STD))
            else:
                if metric == "SNR":
                    print(f"Skipping SNR for datatype {datatype}.")
                    continue
                print(f"Unexpected metric type: {metric}")
                raise ValueError("Wrong Test Metric Type")
    else:
        raise ValueError("Inference evaluation method name wrong!")

    # Create the result dictionary to return
    result = {
        "metrics": metrics_result,
        "datatype": datatype
    }

    return result


# 计算视频切片的metrics
def calculate_metrics_simple(predictions, labels):

    TEST_METRICS = ["MAE","RMSE","MAPE","Pearson"]
    metrics_result = {}
    num_test_samples = len(predictions)
    
    for metric in TEST_METRICS:
        if metric == "MAE":
            MAE_FFT = np.mean(np.abs(predictions - labels))
            MAE_STD = np.std(np.abs(predictions - labels)) / np.sqrt(num_test_samples)
            metrics_result["FFT_MAE"] = (MAE_FFT)
            print("FFT MAE (FFT Label): {0} +/- {1}".format(MAE_FFT, MAE_STD))
        elif metric == "RMSE":
            RMSE_FFT = np.sqrt(np.mean(np.square(predictions - labels)))
            RMSE_STD = np.std(np.square(predictions - labels)) / np.sqrt(num_test_samples)
            metrics_result["FFT_RMSE"] = (RMSE_FFT)
            print("FFT RMSE (FFT Label): {0} +/- {1}".format(RMSE_FFT, RMSE_STD))
        elif metric == "MAPE":
            MAPE_FFT = np.mean(np.abs((predictions - labels) / labels)) * 100
            MAPE_STD = np.std(np.abs((predictions - labels) / labels)) / np.sqrt(num_test_samples) * 100
            metrics_result["FFT_MAPE"] = (MAPE_FFT)
            print("FFT MAPE (FFT Label): {0} +/- {1}".format(MAPE_FFT, MAPE_STD))
        elif metric == "Pearson":
            Pearson_FFT = np.corrcoef(predictions, labels)
            correlation_coefficient = Pearson_FFT[0][1]
            Pearson_STD = np.sqrt((1 - correlation_coefficient**2) / (num_test_samples - 2))
            metrics_result["FFT_Pearson"] = (correlation_coefficient)
            print("FFT Pearson (FFT Label): {0} +/- {1}".format(correlation_coefficient, Pearson_STD))
        else:
            print("FFT SNR (FFT Label): No valid SNR values")

    return metrics_result


# Example usage:

# predictions = {"bvp":np.random.rand(100)}
# labels = np.random.randint(2, size=100)
# result = calculate_metrics(predictions, labels, None, "Spo2")
# print(result["metrics"]["FFT_MAE"])