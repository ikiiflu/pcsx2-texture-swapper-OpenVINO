# PCSX2 Texture Swapper com IA 🎮

Um **texture swapper experimental** que usa inteligência artificial (modelos otimizados para CPU via OpenVINO) para substituir e regenerar texturas de jogos PlayStation 2 emulados no PCSX2. 

> ⚠️ **Aviso**: Este projeto usa modelos otimizados para execução em **CPU apenas**. A qualidade das texturas geradas é limitada. **Você pode (e deve) adaptar o código para usar modelos mais avançados** em GPUs (NVIDIA/AMD) ou versões mais poderosas de modelos. Veja a seção [Customização](#-customização-e-melhorias) para detalhes.

---

## 🌟 Características Principais

- **Dois modos de processamento**:
  - **Com contexto (BLIP + img2img)**: Analisa a textura original via IA, gera descrição automática e cria uma versão transformada mantendo a estrutura
  - **Sem contexto (txt2img)**: Gera texturas completamente novas a partir de palavras-chave aleatórias

- **Interface gráfica intuitiva** (tkinter) com:
  - Seleção da pasta de texturas do jogo (ex: `textures/SLUS-20001`)
  - Controle de parâmetros em tempo real
  - Log detalhado de atividades
  - Temas/presets de prompts customizáveis

- **Processamento em lote**:
  - Modo **total**: Processa todas as texturas
  - Modo **parcial**: Seleciona quantidade específica ou metade + extras aleatórias

- **Otimizado para CPU**:
  - Usa OpenVINO para otimizar modelos Stable Diffusion
  - LCMScheduler para reduzir passos de inferência
  - Ideal para notebooks sem GPU dedicada

- **Integração com PCSX2**:
  - Salva texturas em formato compatível (`replacements/`)
  - Preserva canais alpha de transparência
  - Upscaling automático 2x

---

## 📋 Pré-requisitos

- **Python 3.8+**
- **PCSX2** (versão recente) com suporte a texture replacement
- **Texturas dumped** do jogo (PCSX2 salva automaticamente na subpasta `dumps/` do seu jogo)
- Espaço em disco: ~8GB para modelos + espaço para texturas geradas

### Dependências Python

Instale com pip:

```bash
pip install torch transformers diffusers openvino[onnx] openvino-nightly optimum[intel] pillow
```

> **Para GPU (NVIDIA/CUDA)**: Se você quiser usar GPU em vez de CPU, substitua `torch` por `torch --index-url https://download.pytorch.org/whl/cu118` e remova as dependências de OpenVINO.

---

## 🚀 Instalação

1. **Clone o repositório**:
   ```bash
   git clone [https://github.com/ikiiflu/pcsx2-texture-swapper-OpenVINO.git](https://github.com/ikiiflu/pcsx2-texture-swapper-OpenVINO.git)
   cd pcsx2-texture-swapper-OpenVino
   ```

2. **Crie um ambiente virtual** (recomendado):
   ```bash
   python -m venv venv
   source venv/bin/activate  # No Windows: venv\Scripts\activate
   ```

3. **Instale as dependências**:
   ```bash
   pip install torch transformers diffusers openvino[onnx] openvino-nightly optimum[intel] pillow
   ```
   
   *Ou, crie um arquivo `requirements.txt` com essas dependências e instale com:*
   ```bash
   pip install -r requirements.txt
   ```

4. **Execute o script**:
   ```bash
   python texture_swapper.py
   ```

---

## 📖 Como Usar

### Passo a Passo

1. **No PCSX2**: Configure a extração de texturas
   - Abra: **Configurações → Gráficos**
   - Verifique se o **renderizador gráfico está habilitado**
   - Vá na aba **Substituições de Texturas** (Texture Replacement)
   - Marque: ✓ **Ativar substituição de texturas**
   - Marque: ✓ **Extrair texturas para arquivo**
   - Jogue o jogo por 2-3 minutos. O PCSX2 vai criar automaticamente uma pasta baseada no código do jogo (ex: `SLUS-20001`) dentro da pasta `textures`, contendo as imagens na subpasta `dumps/`.

2. **Execute o script**:
   ```bash
   python texture_swapper.py
   ```

3. **Na interface**:
   - 📁 **Selecione a pasta do jogo**: Navegue até o diretório do PCSX2 e selecione a pasta com o código do jogo (Ex: `textures/SLUS-20001`). *Não selecione a pasta `dumps` diretamente, selecione a pasta pai do código.*
   - 🎯 **Escolha o modo**: Com contexto ou Sem contexto
   - 📝 **Defina um prompt/tema** (se modo com contexto)
   - ⚙️ **Ajuste os parâmetros**:
     - **Steps**: 4-20 (mais = melhor qualidade, mais lento)
     - **Strength**: 0.5-0.9 (intensidade de transformação)
     - **Guidance**: 5-15 (como seguir o prompt)
   - ▶️ **Clique em INICIAR**

### Estrutura de Pastas Esperada

Para que a integração funcione perfeitamente com o PCSX2, a estrutura que o programa usa é esta:

```text
textures/
└── SLUS-20001/          ← 📁 ESTA é a pasta que você deve selecionar na interface!
    ├── dumps/           ← Texturas originais (dumped pelo PCSX2)
    │   ├── texture_001.png
    │   ├── texture_002.png
    │   └── ...
    └── replacements/    ← Texturas geradas (salvas automaticamente pelo script)
        ├── texture_001.png
        ├── texture_002.png
        └── ...
```
*(Nota: `SLUS-20001` é apenas um exemplo. O código da pasta será diferente dependendo da região e do jogo).*

---

## ⚡ Configuração Rápida

O script usa **defaults internos** automaticamente se `prompts.json` e `random_words.json` não existirem. 
Mas se quiser customizar seus próprios temas e palavras, siga a seção abaixo:

---

## ⚙️ Configuração de Presets

O script procura por dois arquivos JSON na mesma pasta:

### `prompts.json`

Define temas/presets de prompts:

```json
[
  {
    "name": "High Fantasy",
    "prompt": "intricate fantasy texture, epic stone runes, detailed medieval art style",
    "negative": "modern, plastic, low quality"
  },
  {
    "name": "Sci-Fi",
    "prompt": "futuristic sci-fi material, metallic surface, neon glows, cyber aesthetic",
    "negative": "organic, natural, earthy"
  },
  {
    "name": "Default",
    "prompt": "seamless game texture, high quality material",
    "negative": ""
  }
]
```

### `random_words.json`

Lista de palavras para modo sem contexto:

```json
[
  "stone", "metal", "wood", "fabric", "marble", "rust", "crystal",
  "sand", "ice", "fire", "water", "abstract", "mosaic", "geometric",
  "organic", "alien", "digital", "gothic"
]
```

Se esses arquivos não existirem, o script usa defaults.

---

## 📊 Modos de Operação

### Modo Com Contexto (Com BLIP)

```
Textura original → BLIP (visão) → Descrição automática
                                     ↓
                              + Seu prompt temático
                                     ↓
                        DreamShaper img2img
                                     ↓
                        Textura transformada
```

**Vantagens**: Mantém estrutura base da textura, mais previsível  
**Desvantagem**: Mais lento (BLIP + modelo grande)

### Modo Sem Contexto (Txt2img)

```
Palavra aleatória → Prompt + Seu tema → DreamShaper txt2img → Textura nova
```

**Vantagens**: Mais rápido, criatividade maior  
**Desvantagem**: Menos controle, texturas podem não combinar com o jogo

---

## ⚡ Performance

### Tempos Esperados (CPU Intel/AMD moderno)

| Modo | Steps | Tempo/Textura |
|------|-------|---------------|
| Txt2img (sem BLIP) | 4 | 30-60s |
| Txt2img (sem BLIP) | 10 | 1-2 min |
| Img2img com BLIP | 4 | 3-5 min |
| Img2img com BLIP | 10 | 5-10 min |

> ⚠️ **Primeira execução**: Modelos são baixados (~2GB) e otimizados (~5-10 min)

---

## 🎨 Qualidade e Limitações

⚠️ **Este projeto usa modelos otimizados para CPU, sacrificando qualidade**:

- DreamShaper 8 é relativamente pequeno (2.1B parâmetros)
- OpenVINO otimiza para CPU, reduzindo precisão
- Redução de passos para performance = artefatos visíveis
- Texturas de 512×512 podem parecer pixeladas em jogos com alta resolução

### Exemplos de Limitações

- Texturas podem ter padrões estranhos
- Detalhes finos podem ser distorcidos
- Alguns prompts geram resultados irreais

**Isso é esperado e aceitável para um projeto experimental em CPU.**

---

## 🔧 Customização e Melhorias

### ⭐ Como Melhorar a Qualidade

**Se você tem GPU**, reescreva o código para usar:

#### 1. **Modelos Melhores** (Stable Diffusion 3, FLUX, etc)

```python
# Exemplo: Usar Stable Diffusion 3 com GPU
from diffusers import StableDiffusion3Pipeline
import torch

pipeline = StableDiffusion3Pipeline.from_pretrained(
    "stabilityai/stable-diffusion-3-medium-diffusers",
    torch_dtype=torch.float16
).to("cuda")
```

#### 2. **Schedulers Mais Sofisticados**

```python
# Ao invés de LCMScheduler
from diffusers import EulerDiscreteScheduler, DPMSolverMultistepScheduler

pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
    pipeline.scheduler.config
)
```

#### 3. **Upscaling Real**

Integre modelos de upscaling como Real-ESRGAN ou BSRGAN:

```python
from RealESRGAN import RealESRGAN

upscaler = RealESRGAN(scale=4)
high_res_texture = upscaler.predict(generated_texture)
```

#### 4. **Controlnets para Mais Controle**

```python
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
import torch

controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-canny"
)
pipeline = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", 
    controlnet=controlnet
)
```

#### 5. **Lote com Batch Processing**

```python
# Processar múltiplas texturas em paralelo na GPU
from concurrent.futures import ThreadPoolExecutor

with ThreadPoolExecutor(max_workers=4) as executor:
    futures = [executor.submit(process_texture, tex) for tex in textures]
```

### Sugestões de Fork/Adaptações

1. **`texture-swapper-gpu`**: Versão NVIDIA CUDA com Stable Diffusion 3
2. **`texture-swapper-amd`**: Versão AMD ROCm

---

## 🐛 Troubleshooting

### Erro: `ModuleNotFoundError: No module named 'optimum.intel'`

```bash
pip install --upgrade optimum[intel] openvino openvino-nightly
```

### Erro: `OutOfMemoryError` ou muito lento

- Reduza `steps` (mínimo 4)
- Use `strength` menor (0.5-0.6)
- Feche outros programas
- Em GPU: reduza `torch.cuda.empty_cache()`

### PCSX2 não reconhece as texturas

Verifique:
- A pasta `replacements/` contida no diretório do seu jogo (ex: `textures/SLUS-20001/replacements/`) possui os arquivos `.png`.
- Config → Graphics → Texture Replacement está ativado.
- O caminho no emulador aponta para a pasta global de texturas, e a nomenclatura dos arquivos bate perfeitamente com os que estavam em `dumps/`.

### Texturas ficam muito estranhas

- Aumente `guidance` (mais fidelidade ao prompt)
- Reduza `strength` (menos transformação)
- Use prompts mais específicos (ex: "stone brick" vs "rock")
- Tente o outro modo (com/sem contexto)

---

## 📝 Exemplos de Prompts Bons

```
✓ "seamless medieval stone wall, high quality texture, detailed cracks"
✓ "sci-fi metal panel, brushed aluminum, futuristic tech"
✓ "fantasy forest floor, moss texture, organic materials"
✗ "cool texture" (muito vago)
✗ "photorealistic face" (modelo foi treinado para texturas, não rostos)
```

---

## 📄 Licença

Este projeto é fornecido **como está**, para fins educacionais e experimentais.

**Modelos usados**:
- DreamShaper 8 por Lykon (Stable Diffusion fine-tuned) — CC BY-NC-SA 4.0
- BLIP por Salesforce Research — BSD 3-Clause
- OpenVINO by Intel — Apache 2.0

Respeite as licenças ao adaptar/distribuir versões derivadas.

---

## 🤝 Contribuições

Sinta-se livre para:
- Reportar bugs via Issues
- Sugerir melhorias
- Enviar PRs com otimizações ou novos modos
- **Criar versões com GPU** e compartilhar!

---

## 💡 Inspiração e Referências

- [OpenVINO Intel](https://github.com/openvinotoolkit/openvino)
- [Diffusers by HuggingFace](https://github.com/huggingface/diffusers)
- [PCSX2 Texture Replacement](https://github.com/PCSX2/pcsx2/wiki/Texture-Replacement)
- [Stable Diffusion](https://github.com/CompVis/stable-diffusion)

---

## 🎯 Roadmap

- [ ] Suporte para GPU (CUDA, ROCm)
- [ ] Interface web (FastAPI + React)
- [ ] Batch processing paralelo
- [ ] Integração com Real-ESRGAN para upscaling
- [ ] ControlNet para mais controle
- [ ] Preview em tempo real das texturas geradas
- [ ] Sistema de undo/comparação antes/depois

---

## ⚠️ Aviso Legal

Este projeto é **experimental**. Não há garantias de qualidade ou funcionalidade. Use por sua conta e risco.

**Texturas geradas por IA podem**:
- Violar direitos autorais (use para projeto pessoal/modificado)
- Não corresponder ao estilo original do jogo
- Ser inapropriadas para compartilhamento público

---

**❤️ PS2**

*Última atualização: abril de 2026*
