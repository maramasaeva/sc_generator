from transformers import PretrainedConfig


PATH = "."

class MusiLingoConfig(PretrainedConfig):
    model_type = "musilingo"
    is_encoder_decoder = True
    def __init__(self,
                    mert_model = "m-a-p/MERT-v1-330M",
                    llama_model = f'lmsys/vicuna-7b-delta-v0',
                    prompt_path = "",
                    prompt_template = '###Human: {} ###Assistant: ',
                max_txt_len = 32,
                end_sym = '\n',
                low_resource = False,
                device_8bit = 0,
                # linear_ckpt_path = "",
                    **kwargs):
        self.mert_model = mert_model
        self.llama_model = llama_model
        self.prompt_path = prompt_path
        self.prompt_template = prompt_template
        self.max_txt_len = max_txt_len
        self.end_sym = end_sym
        self.low_resource = low_resource
        self.device_8bit = device_8bit
        # self.linear_ckpt_path = linear_ckpt_path
        super().__init__(**kwargs)