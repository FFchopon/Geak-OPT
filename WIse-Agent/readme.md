- tool_script
    - profile.py - 使用性能测试输入，生成指定算子的profile信息
    - verify_test.py - 用于运行指定算子，评估其三个指标
    - verify_profile.py - profile.py + verify_test.py 

- optimize.py - 用于优化指定算子
    - step1: 生成优化策略（仅执行1次）
    - step2: 基于优化策略，优化算子实现（执行多次）
    - 参数说明
        - NCU_FULL_REPORT: 0/1，0使用-f的四张表，1使用一张总表（包含在四张表中）

- optimize_workflow.py - 用于优化指定算子
    - 加入了经验教训的积累。
    - 有超过1.05x的，直接进入下一轮。加入一个中间经验分析LLM，提供优化策略以及优化前后profiling对比信息 → 提炼优化经验
    - 没有超过1.05x的，整体Fail率低于50%，有改进后性能没提升或显著下降的。使用中间经验分析LLM，提供优化策略以及优化前后profiling对比信息 → 提炼优化教训