import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import torch
import tensorflow
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from torchvision import transforms
from transformers import CLIPProcessor, CLIPModel
import warnings
warnings.filterwarnings('ignore')

if __name__ == '__main__':

    """ Verificando o Device """

    if torch.cuda.is_available():
        print('Número de GPUs:', torch.cuda.device_count())
        print('Modelo GPU:', torch.cuda.get_device_name(0))
        print('Total Memória [GB] da GPU:', torch.cuda.get_device_properties(0).total_memory / 1e9)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)

    """ Carregando a Imagem """

    # Imagem
    imagem = Image.open("dados/imagem.jpg")

    # Plot da imagem
    plt.imshow(imagem)
    plt.axis('off')
    plt.show()

    """ Pré-Processamento """

    # Função para transformar a imagem em tensor
    transt = transforms.ToTensor()

    # Aplica a função
    img = transt(imagem)

    # Adiciona dimensão de lote (batch) e mudar a dimensão de canais de cores
    patches = img.data.unfold(0, 3, 3)

    # Divide a imagem em patches na dimensão de altura
    patch = 256
    new_patches = patches.unfold(1, patch, patch)

    # Divide a imagem em patches na dimensão de largura
    new_patches_1 = new_patches.unfold(2, patch, patch)

    """ Modelagem com OpenAI CLIP """

    # Define o modelo
    modelo_id = "openai/clip-vit-base-patch32"

    # Carrega o modelo
    modelo = CLIPModel.from_pretrained(modelo_id)

    # Carrega o processador do modelo
    processor = CLIPProcessor.from_pretrained(modelo_id)

    # Manda o modelo para o device
    modelo.to(device)

    """ Usando o Modelo Para Extrair os Scores (Classificações) """

    # Hiperparâmetros
    window = 6
    stride = 1

    # Variáveis de métricas
    scores = torch.zeros(new_patches_1.shape[1], new_patches_1.shape[2])
    runs = torch.ones(new_patches_1.shape[1], new_patches_1.shape[2])

    # Iniciar loop principal para percorrer o eixo Y dos patches
    for Y in range(0, new_patches_1.shape[1] - window + 1, stride):

        # Iniciar loop secundário para percorrer o eixo X dos patches
        for X in range(0, new_patches_1.shape[2] - window + 1, stride):

            # Inicializar um tensor vazio para armazenar os patches combinados de maior tamanho
            big_patch = torch.zeros(patch * window, patch * window, 3)

            # Extrair uma janela de patches da posição atual (Y, X)
            patch_batch = new_patches_1[0, Y:Y + window, X:X + window]

            # Iterar sobre a dimensão y da janela de patches
            for y in range(window):

                # Iterar sobre a dimensão x da janela de patches
                for x in range(window):
                    # Reordenar e colocar cada patch pequeno na posição correta dentro do big_patch
                    big_patch[y * patch:(y + 1) * patch, x * patch:(x + 1) * patch, :] = patch_batch[y, x].permute(1, 2,
                                                                                                                   0)

            # Processar o big_patch para transformá-lo em um tensor compatível com PyTorch
            # e movê-lo para o device
            inputs = processor(images=big_patch, return_tensors="pt", text="an animal", padding=True).to(device)

            # Usar o modelo para obter um score para o big_patch
            score = modelo(**inputs).logits_per_image.item()

            # Atualizar a matriz de scores com o score obtido para a janela atual
            scores[Y:Y + window, X:X + window] += score

            # Incrementar a contagem na matriz runs para cada posição na janela atual
            runs[Y:Y + window, X:X + window] += 1

    """ Preparando as Bounding Boxes Para Visualização a Partirs dos Scores """

    # Scores divididos pelos runs
    scores /= runs

    # Clip dos scores
    # Essa operação garante que todos os valores em scores sejam não negativos e que
    # valores abaixo da média original sejam definidos como zero.
    scores = np.clip(scores - scores.mean(), 0, np.inf)

    # Normaliza os scores
    scores = (scores - scores.min()) / (scores.max() - scores.min())

    # Transforma os patches em tensor
    adj_patches = new_patches_1.squeeze(0).permute(3, 4, 2, 0, 1)

    # Multiplica os patches pelos scores
    adj_patches = adj_patches * scores

    # Rotaciona os patches para visualizar
    adj_patches = adj_patches.permute(3, 4, 2, 0, 1)

    # Visualizando todas as bounding boxes

    # X e Y
    Y = adj_patches.shape[0]
    X = adj_patches.shape[1]

    # Plot
    fig, ax = plt.subplots(Y, X, figsize=(X * .5, Y * .5))
    for y in range(Y):
        for x in range(X):
            ax[y, x].imshow(adj_patches[y, x].permute(1, 2, 0))
            ax[y, x].axis("off")
            ax[y, x].set_aspect('equal')
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig('dados/imagem_bbox.png')
    plt.show()

    """ Imagem Final com a Bounding Box de Maior Probabilidade na Classificação """

    # Somente scores maiores que 0.50 serão considerados detecção
    detection = scores > 0.5

    # Coordenadas mínimas de y para a bbox
    y_min, y_max = (np.nonzero(detection)[:, 0].min().item(), np.nonzero(detection)[:, 0].max().item() + 1)

    # Coordenadas máximas de x para a bbox
    x_min, x_max = (np.nonzero(detection)[:, 1].min().item(), np.nonzero(detection)[:, 1].max().item() + 1)

    # Incluímos o pacth
    y_min *= patch
    y_max *= patch
    x_min *= patch
    x_max *= patch

    # Altura e largura da bbox
    height = y_max - y_min
    width = x_max - x_min

    # Move a dimensão do canal de cor para o final
    image = np.moveaxis(img.data.numpy(), 0, -1)

    # Imprime a imagem e a bounding box
    import matplotlib.patches as patches

    # Figura
    fig, ax = plt.subplots(figsize=(Y * 0.5, X * 0.5))

    # Imagem original
    ax.imshow(image)

    # Cria o retângulo da bbox
    rect = patches.Rectangle((x_min, y_min), width, height, linewidth=3, edgecolor='#0800ff', facecolor='none')

    # Adiciona o patch ao plot
    ax.add_patch(rect)

    plt.savefig('dados/imagem_final.png')
    plt.show()