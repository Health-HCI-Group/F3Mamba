import torch

def test_cuda():
    print(f"PyTorch version: {torch.__version__}")
    
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")
    
    if cuda_available:
        print(f"CUDA version: {torch.version.cuda}")
        device_count = torch.cuda.device_count()
        print(f"Number of GPUs: {device_count}")
        for i in range(device_count):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        
        # Test a simple tensor operation on GPU
        try:
            x = torch.tensor([1.0, 2.0, 3.0]).cuda()
            y = torch.tensor([4.0, 5.0, 6.0]).cuda()
            z = x + y
            print(f"Tensor operation on GPU successful. Output: {z.cpu().numpy()}")
        except Exception as e:
            print(f"Error during tensor operation on GPU: {e}")
    else:
        print("CUDA is not available. Please check your PyTorch and CUDA installation.")

if __name__ == "__main__":
    test_cuda()
