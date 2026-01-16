from agents.GaAgent import GaAgent
from models.OpenAI import OpenAIModel
from models.Gemini import GeminiModel
from models.Claude import ClaudeModel
from models.DeepSeek import DeepSeekModel
from models.Qwen import QwenModel
from dataloaders.TritonBench import TritonBench
from args_config import load_config


def main():
    args = load_config("configs/tritonbench_gaagent_config.yaml")

    # setup LLM model
    model_backend = getattr(args, "model_backend", "claude")
    if model_backend == "claude":
        model = ClaudeModel(api_key=args.api_key, model_id=args.model_id)
    elif model_backend == "openai":
        model = OpenAIModel(api_key=args.api_key, model_id=args.model_id)
    elif model_backend == "gemini":
        model = GeminiModel(api_key=args.api_key, model_id=args.model_id)
    elif model_backend == "deepseek":
        model = DeepSeekModel(api_key=args.api_key, model_id=args.model_id)
    elif model_backend == "qwen":
        model = QwenModel(api_key=args.api_key, model_id=args.model_id)
    else:
        raise ValueError(f"Unsupported model_backend: {model_backend}")

    # setup dataset
    dataset = TritonBench(statis_path=args.statis_path, 
                          py_folder=args.py_folder, 
                          instruction_path=args.instruction_path, 
                          py_interpreter=args.py_interpreter, 
                          golden_metrics=args.golden_metrics,
                          perf_ref_folder=args.perf_ref_folder,
                          perf_G_path=args.perf_G_path,
                          result_path=args.result_path)

    # setup agent
    agent = GaAgent(model=model, dataset=dataset, corpus_path=args.corpus_path, mem_file=args.mem_file, descendant_num=args.descendant_num)

    # run the agent
    agent.run(output_path=args.output_path, 
              multi_thread=args.multi_thread, 
              iteration_num=args.max_iteration, 
              temperature=args.temperature, 
              datalen=args.datalen,
              gpu_id=args.gpu_id,
              start_iter=args.start_iter,
              ancestor_num=args.ancestor_num,
              descendant_num=args.descendant_num,
              descendant_debug=args.descendant_debug,
              target_gpu=args.target_gpu,
              profiling=args.profiling,
              start_idx=args.start_idx)


if __name__ == "__main__":
    main()