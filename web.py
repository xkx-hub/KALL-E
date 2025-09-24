
import os
import traceback
import gradio as gr
import torch

from loguru import logger
from infer import infer_tools

# Make einx happy
os.environ["EINX_FILTER_TRACEBACK"] = "false"

HEADER_MD = f"""# 

{("KALL-E")}  

"""

TEXTBOX_PLACEHOLDER = ("Put your text here.")
SPACE_IMPORTED = False


@torch.inference_mode()
def inference(
    reference_audio,
    reference_text,
    refined_text,
    enable_reference_audio,
):

    try:   
        print(f'refined_text: {refined_text}')
        print(f'reference_text: {reference_text}')
        if enable_reference_audio:

            audio = infer_helper.infer(refined_text,reference_text,reference_audio)
        else:
            audio = infer_helper.infer(refined_text)

        result = None, (16000, audio), "no error"
        # return result[0], result[1], result[2]
        return result[1], result[2]
    except Exception as e:
        logger.error(f"发生错误: {e}", exc_info=True)  # 记录异常信息和堆栈
        logger.error("错误详细信息:\n" + traceback.format_exc())  # 记录完整的异常堆栈信息
        return None, f"error:{e}"


n_audios = 1

global_audio_list = []
global_error_list = []



def build_app():
    global infer_helper
    with gr.Blocks(theme=gr.themes.Base()) as app:
        gr.Markdown(HEADER_MD)

        # Use light theme by default
        app.load(
            None,
            None,
            js="() => {const params = new URLSearchParams(window.location.search);if (!params.has('__theme')) {params.set('__theme', '%s');window.location.search = params.toString();}}"
            % "light",
        )

        # Inference
        with gr.Row():
            with gr.Column(scale=3):
                text = gr.Textbox(
                    label=("Input Text"), placeholder=TEXTBOX_PLACEHOLDER, lines=5,value="KALL-E is a text-to-speech system that predicts continuous speech representations using a single autoregressive language model."
                )

                with gr.Row():

                    with gr.Tab(label=("Reference Audio")):
                        gr.Markdown(
                            (
                                "[optional] 5 to 10 seconds of reference audio, useful for specifying speaker."
                            )
                        )

                        enable_reference_audio = gr.Checkbox(
                            label=("Enable Reference Audio"),
                        )

                        reference_audio = gr.Audio(
                            label=("Reference Audio"),
                            type="filepath",
                            interactive=True,
                        )
                        reference_text = gr.Textbox(
                            label=("Reference Text"), placeholder=("Enter the transcribtion of the reference audio"), lines=5
                        )

            with gr.Column(scale=3):
                with gr.Row():
                    error = gr.Text(
                        label=("Error Message"),
                        visible=True,
                    )
                    global_error_list.append(error)
                with gr.Row():
                    audio = gr.Audio(
                        label=("Generated Audio"),
                        type="numpy",
                        interactive=False,
                        visible=True,
                    )
                    global_audio_list.append(audio)

                with gr.Row():
                    with gr.Column(scale=3):
                        generate = gr.Button(
                            value="\U0001F3A7 " + ("Generate"), variant="primary"
                        )

        generate.click(
            inference,
            [
                reference_audio,
                reference_text,
                text,
                enable_reference_audio,

            ],
            [global_audio_list[0], global_error_list[0]]
        )

    return app


if __name__ == "__main__":

    global infer_helper

    infer_helper = infer_tools()

    logger.info("Launching the web UI...")

    app = build_app()
    app.launch(show_api=False,share=True, server_name= "0.0.0.0", server_port = 7861)