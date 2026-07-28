from transformers import AutoTokenizer
t=AutoTokenizer.from_pretrained("/Users/eric/models/Nanbeige/Nanbeige4.2-3B", trust_remote_code=True)
conv=[{"role":"system","content":"You are helpful."},
      {"role":"user","content":"Q1"},
      {"role":"assistant","content":"<think>reasoning one</think>Answer one"},
      {"role":"user","content":"Q2"}]
print("### preserve_thinking default ###")
print(t.apply_chat_template(conv, add_generation_prompt=True, tokenize=False))
print("### preserve_thinking=False ###")
print(t.apply_chat_template(conv, add_generation_prompt=True, tokenize=False, preserve_thinking=False))
tools=[{"type":"function","function":{"name":"get_weather","description":"w","parameters":{"type":"object","properties":{"city":{"type":"string"}}}}}]
tc=[{"role":"user","content":"weather in SF?"},
    {"role":"assistant","content":"","tool_calls":[{"type":"function","function":{"name":"get_weather","arguments":{"city":"SF"}}}]},
    {"role":"tool","content":"18C"},
    {"role":"tool","content":"sunny"},
    {"role":"user","content":"thanks"}]
print("### tools (xml default) ###")
print(t.apply_chat_template(tc, tools=tools, add_generation_prompt=True, tokenize=False))
