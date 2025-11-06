import uuid
import gradio as gr
from graph_entity.graph import graph_builder, thread_id_genrater, message, reset

with gr.Blocks(theme=gr.themes.Default(primary_hue="emerald")) as demo:
    gr.Markdown("## Personal Agro Assistant AI Supporter ")
    thread = gr.State(thread_id_genrater())
    
    with gr.Row():
        chatbot = gr.Chatbot(label="Agro Assistant AI", height=300, type="messages")
        
    with gr.Group():
        with gr.Row():
            chat_query = gr.Textbox(show_label=False, placeholder="Your query to your Agro Assistant AI system?")
            
    with gr.Row():
        reset_button = gr.Button("Reset", variant="stop")
        go_button = gr.Button("Go!", variant="primary")

    chat_query.submit(message, [chat_query, chatbot, thread], [chatbot])
    go_button.click(message, [chat_query, chatbot, thread], [chatbot])
    reset_button.click(reset, [], [chat_query, chatbot, thread])
    
demo.launch(share=True)

# How to address  Fall Army Worm disease in Maize?
# if __name__ == "__main__":
#     main()
