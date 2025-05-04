import random
import gradio as gr
from sample import (arg_parse, 
                    sampling,
                    load_font_diffuser_pipeline)

def run_fontdiffuer(source_image, 
                    character, 
                    reference_image,
                    shading_image,
                    background_image,
                    sampling_step,
                    guidance_scale,
                    batch_size):
    args = arg_parse()
    args.demo = True
    args.ckpt_dir = 'ckpt'
    args.character_input = False if source_image is not None else True
    args.content_character = character if character else None
    args.num_inference_steps = sampling_step
    args.guidance_scale = guidance_scale
    args.batch_size = batch_size
    args.seed = random.randint(0, 10000)
    pipe = load_font_diffuser_pipeline(args=args)
    out_image = sampling(
        args=args,
        pipe=pipe,
        content_image=source_image,
        style_image=reference_image,
        shading_image=shading_image,
        background_image=background_image)
    return out_image

if __name__ == '__main__':
    args = arg_parse()
    args.demo = True
    args.ckpt_dir = 'ckpt'

    pipe = load_font_diffuser_pipeline(args=args)

    with gr.Blocks() as demo:
        with gr.Row():
            with gr.Column(scale=1):
                gr.HTML("""
                    <div style="text-align: center; max-width: 1200px; margin: 20px auto;">
                    <h1 style="font-weight: 900; font-size: 3rem; margin: 0rem">
                        FontDiffuser
                    </h1>
                    <h2 style="font-weight: 450; font-size: 1rem; margin: 0rem">
                        <a href="https://yeungchenwa.github.io/"">Zhenhua Yang</a>, 
                        <a href="https://scholar.google.com/citations?user=6zNgcjAAAAAJ&hl=zh-CN&oi=ao"">Dezhi Peng</a>, 
                        <a href="https://github.com/kyxscut"">Yuxin Kong</a>, 
                        <a href="https://github.com/ZZXF11"">Yuyi Zhang</a>, 
                        <a href="https://scholar.google.com/citations?user=IpmnLFcAAAAJ&hl=zh-CN&oi=ao"">Cong Yao</a>, 
                        <a href="http://www.dlvc-lab.net/lianwen/Index.html"">Lianwen Jin</a>†
                    </h2>
                    <h2 style="font-weight: 450; font-size: 1rem; margin: 0rem">
                        <strong>South China University of Technology</strong>, Alibaba DAMO Academy
                    </h2>
                    <h3 style="font-weight: 450; font-size: 1rem; margin: 0rem"> 
                    [<a href="https://arxiv.org/abs/2312.12142" style="color:blue;">arXiv</a>] 
                    [<a href="https://yeungchenwa.github.io/fontdiffuser-homepage/" style="color:green;">Homepage</a>]
                    [<a href="https://github.com/yeungchenwa/FontDiffuser" style="color:green;">Github</a>]
                    </h3>
                    <h2 style="text-align: left; font-weight: 600; font-size: 1rem; margin-top: 0.5rem; margin-bottom: 0.5rem">
                    1. We propose FontDiffuser, enhanced with shading and background style migration, for generating unseen characters and styles, including cross-lingual generation (e.g., Chinese to Korean).
                    </h2>
                    <h2 style="text-align: left; font-weight: 600; font-size: 1rem; margin-top: 0.5rem; margin-bottom: 0.5rem">
                    2. FontDiffuser excels in generating complex characters with detailed shading and background styles, achieving state-of-the-art performance.
                    </h2>
                    </div>
                    """)
                gr.Image('figures/result_vis.png')
                gr.Image('figures/demo_tips.png')
            with gr.Column(scale=1):
                with gr.Row():
                    source_image = gr.Image(width=320, label='[Option 1] Source Image', image_mode='RGB', type='pil')
                    reference_image = gr.Image(width=320, label='Reference Image', image_mode='RGB', type='pil')
                with gr.Row():
                    shading_image = gr.Image(width=320, label='Shading Image', image_mode='RGB', type='pil')
                    background_image = gr.Image(width=320, label='Background Image', image_mode='RGB', type='pil')
                with gr.Row():
                    character = gr.Textbox(value='隆', label='[Option 2] Source Character')
                with gr.Row():
                    fontdiffuer_output_image = gr.Image(height=200, label="FontDiffuser Output Image", image_mode='RGB', type='pil')

                sampling_step = gr.Slider(20, 50, value=20, step=10, 
                                          label="Sampling Step", info="The sampling step by FontDiffuser.")
                guidance_scale = gr.Slider(1, 12, value=7.5, step=0.5, 
                                           label="Scale of Classifier-free Guidance", 
                                           info="The scale used for classifier-free guidance sampling")
                batch_size = gr.Slider(1, 4, value=1, step=1, 
                                       label="Batch Size", info="The number of images to be sampled.")

                FontDiffuser = gr.Button('Run FontDiffuser')
                gr.Markdown("## <font color=#008000, size=6>Examples that You Can Choose Below⬇️</font>")
        with gr.Row():
            gr.Markdown("## Examples")
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("## Example 1️⃣: Source Image, Reference Image, Shading, Background")
                gr.Markdown("### Provide source, reference, shading, and background images to try our demo!")
                gr.Examples(
                    examples=[
                        ['figures/source_imgs/source_灨.jpg', 'figures/ref_imgs/ref_籍.jpg', 'figures/shading_imgs/874.png', 'figures/background_imgs/1.png'],
                        ['figures/source_imgs/source_鑻.jpg', 'figures/ref_imgs/ref_鹰.jpg', 'figures/shading_imgs/875.png', 'figures/background_imgs/2.png'],
                        ['figures/source_imgs/source_鑫.jpg', 'figures/ref_imgs/ref_壤.jpg', 'figures/shading_imgs/876.png', 'figures/background_imgs/3.png'],
                        ['figures/source_imgs/source_釅.jpg', 'figures/ref_imgs/ref_雕.jpg', 'figures/shading_imgs/877.png', 'figures/background_imgs/4.png']
                    ],
                    inputs=[source_image, reference_image, shading_image, background_image]
                )
            with gr.Column(scale=1):
                gr.Markdown("## Example 2️⃣: Character, Reference Image, Shading, Background")
                gr.Markdown("### Provide a content character, reference, shading, and background images to try our demo!")
                gr.Examples(
                    examples=[
                        ['龍', 'figures/ref_imgs/ref_鷢.jpg', 'figures/shading_imgs/878.png', 'figures/background_imgs/5.png'],
                        ['轉', 'figures/ref_imgs/ref_鲸.jpg', 'figures/shading_imgs/869.png', 'figures/background_imgs/906.png'],
                        ['懭', 'figures/ref_imgs/ref_籍_1.jpg', 'figures/shading_imgs/870.png', 'figures/background_imgs/907.png'],
                        ['識', 'figures/ref_imgs/ref_鞣.jpg', 'figures/shading_imgs/871.png', 'figures/background_imgs/908.png']
                    ],
                    inputs=[character, reference_image, shading_image, background_image]
                )
            with gr.Column(scale=1):
                gr.Markdown("## Example 3️⃣: Reference Image, Shading, Background")
                gr.Markdown("### Provide reference, shading, and background images; upload your own source image or choose a character!")
                gr.Examples(
                    examples=[
                        ['figures/ref_imgs/ref_闡.jpg', 'figures/shading_imgs/872.png', 'figures/background_imgs/909.png'],
                        ['figures/ref_imgs/ref_雕.jpg', 'figures/shading_imgs/873.png', 'figures/background_imgs/910.png'],
                        ['figures/ref_imgs/ref_豄.jpg', 'figures/shading_imgs/874.png', 'figures/background_imgs/911.png'],
                        ['figures/ref_imgs/ref_馨.jpg', 'figures/shading_imgs/875.png', 'figures/background_imgs/1.png']
                    ],
                    examples_per_page=20,
                    inputs=[reference_image, shading_image, background_image]
                )
        FontDiffuser.click(
            fn=run_fontdiffuer,
            inputs=[
                source_image, 
                character, 
                reference_image,
                shading_image,
                background_image,
                sampling_step,
                guidance_scale,
                batch_size
            ],
            outputs=fontdiffuer_output_image)
    demo.launch(server_name="0.0.0.0", debug=True)