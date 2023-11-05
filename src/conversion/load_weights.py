"""
Load GPT-2 weights from Huggingface.
"""

from transformers import GPT2LMHeadModel

def load_gpt2_weights(model_name):
    model = GPT2LMHeadModel.from_pretrained(model_name)

    # iterate over all model parameters
    for name, param in model.named_parameters():
        # check if we have those parameters in our model
        if name in model.state_dict():
            # load the parameter
            param.data = model.state_dict()[name].data

            print(name, param.data.shape)
        else:
            print(f'Parameter {name} not found in model state dict')
    
    return model

# Usage
model_name = 'gpt2'  # or 'gpt2-medium', 'gpt2-large', 'gpt2-xl'
model = load_gpt2_weights(model_name)
print(model)