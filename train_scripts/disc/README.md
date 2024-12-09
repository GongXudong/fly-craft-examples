# Discoutinuity

## Note

1. 环境使用scaled版本

```python
    import flycraft
    from utils_my.sb3.my_wrappers import ScaledObservationWrapper, ScaledActionWrapper
    env = gym.make("FlyCraft-v0")
    env = ScaledActionWrapper(ScaledObservationWrapper(env))
```