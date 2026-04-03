import gradio as gr
import numpy as np
import assets

from src.config import InferenceConfig
from src.onnx import ClipEBCOnnx

model = ClipEBCOnnx(InferenceConfig(model_path="assets/CLIP_EBC_nwpu_rmse_onnx.onnx"))


def predict_crowd(image):
    if image is None:
        return "이미지를 업로드하세요.", None, None

    counts, heats, dots = model.predict_dense_dot([image])
    return f"예측된 군중 수: {counts[0]:.1f}명", heats[0], dots[0]


with gr.Blocks(title="CLIP-EBC Crowd Counter") as app:
    gr.Markdown("# CLIP-EBC Crowd Counter")
    gr.Markdown("이미지를 업로드하여 군중 수를 예측하고 시각화합니다.")

    with gr.Row():
        input_image = gr.Image(type="numpy", label="입력 이미지")

    with gr.Row():
        predict_btn = gr.Button("예측", variant="primary")

    with gr.Row():
        count_text = gr.Textbox(label="예측 결과")

    with gr.Row():
        with gr.Column():
            density_output = gr.Image(label="밀도 맵")
        with gr.Column():
            dots_output = gr.Image(label="점 시각화")

    predict_btn.click(
        fn=predict_crowd,
        inputs=input_image,
        outputs=[count_text, density_output, dots_output],
    )

if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7860, share=False)
