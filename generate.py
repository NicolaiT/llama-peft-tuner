import torch
import transformers
from finetune_peft import get_peft_config, PEFTArguments
from peft import get_peft_model

model_path = 'models/llama-7b-hf'
peft_path = 'models/alpaca-llama-7b-peft/params.p'

torch.set_default_tensor_type(torch.cuda.HalfTensor)
model = transformers.LlamaForCausalLM.from_pretrained(model_path)
peft_config = get_peft_config(peft_args=PEFTArguments(peft_mode="lora"))
model = get_peft_model(model, peft_config)
model.load_state_dict(torch.load(peft_path), strict=False)
torch.set_default_tensor_type(torch.cuda.FloatTensor)

tokenizer = transformers.LlamaTokenizer.from_pretrained(model_path)
batch = tokenizer("""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
I want to teach on week 22 and 26. This is for lectures. As for the examinations, I would like to have them between 15 and 19. There will be 22 students for both activities.
There will also be two lab sessions on week 25, 26 and 27 of 11 students. 

### Input:


### Response:""", return_tensors="pt")

with torch.no_grad():
    out = model.generate(
        input_ids=batch["input_ids"],
        attention_mask=torch.ones_like(batch["input_ids"]),
        max_length=512,
    )
print(tokenizer.decode(out[0]))
