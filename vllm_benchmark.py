from vllm import LLM, SamplingParams
import time

llm = LLM(
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    dtype="float16",
    max_model_len=512,
    gpu_memory_utilization=0.8
)

sampling_params = SamplingParams(temperature=0.7, max_tokens=80)
questions = [
    "<s>[INST] I want to buy a used car, what is the process? [/INST]",
    "<s>[INST] How do I list my apartment for sale? [/INST]",
    "<s>[INST] What documents do I need to sell my car? [/INST]",
    "<s>[INST] I am looking for a 3 bedroom apartment in Tel Aviv [/INST]",
    "<s>[INST] Can I negotiate the price with the seller? [/INST]",
]

start = time.time()
outputs = llm.generate(questions, sampling_params)
elapsed = time.time() - start

total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
print(f"Tokens/sec: {total_tokens/elapsed:.1f}")
print(f"Improvement: {(total_tokens/elapsed)/13.0:.1f}x over baseline")
