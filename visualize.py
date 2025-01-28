from torchviz import make_dot
import torch
def visualize_model(model, input_size):
    dummy_input = torch.randn(1, input_size)  # Örnek bir giriş verisi
    output = model(dummy_input)  # Model çıktısını al
    dot = make_dot(output, params=dict(model.named_parameters()))
    dot.render("model_architecture", format="png")  # PNG olarak kaydet
    print("Model mimarisi görselleştirildi ve 'model_architecture.png' olarak kaydedildi.")
