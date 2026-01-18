import torch
import torch.nn.functional as F


from dormant_callback import _get_dormant_metrics 

def test_metrics():
    print("=== 开始单元测试 ===")
    
    batch_size = 10
    hidden_dim = 100
    
    # 场景 1: 模拟完全活跃的网络 (所有值都很大，经过Tanh后接近1或-1)
    # 预期: 休眠率应接近 0%
    activations_active = {
        "layer1": torch.tanh(torch.randn(batch_size, hidden_dim) * 100) # 乘以100确保饱和
    }
    zero, dormant = _get_dormant_metrics(activations_active, tau=0.025, skip_last=False)
    print(f"场景1 [全活跃]: 休眠率 {dormant:.2f}% (预期接近 0%)")
    assert dormant < 5.0, "错误：活跃网络被误判为休眠"

    # 场景 2: 模拟部分休眠 (人为制造 50% 的神经元输出为 0)
    # 预期: 休眠率应为 50%
    fake_output = torch.randn(batch_size, hidden_dim) * 10 # 活跃部分
    fake_output[:, 50:] = 0.0 # 后50个神经元强行置零
    activations_half = {
        "layer1": torch.tanh(fake_output)
    }
    zero, dormant = _get_dormant_metrics(activations_half, tau=0.025, skip_last=False)
    print(f"场景2 [一半死]: 休眠率 {dormant:.2f}% (预期 50%)")
    assert abs(dormant - 50.0) < 1.0, "错误：休眠比例计算不准"

    # 场景 3: 测试 skip_last 功能
    # 假设有两层，我们要求 skip_last=True
    activations_multi = {
        "layer1": torch.tanh(torch.randn(batch_size, hidden_dim) * 100), # 活跃
        "layer2": torch.zeros(batch_size, 1) # 全死 (模拟输出层)
    }
    # 如果 skip_last 工作正常，layer2 应该被忽略，结果应该只看 layer1 (活跃)
    _, dormant_skip = _get_dormant_metrics(activations_multi, tau=0.025, skip_last=True)
    print(f"场景3 [Skip Last]: 休眠率 {dormant_skip:.2f}% (预期 0%, 因为死掉的layer2被跳过了)")
    assert dormant_skip < 5.0, "错误：最后一层没有被正确跳过"

    print("=== 测试通过 ✅ ===")

# 运行测试
if __name__ == "__main__":
    # 请确保 _get_dormant_metrics 在作用域内
    test_metrics()