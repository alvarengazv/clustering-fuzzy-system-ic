<a name="readme-topo"></a>

<h1 align='center'>
  Trabalho Intermediário - Agrupamento e Sistemas Fuzzy
</h1>

<div align='center'>

[![SO][Ubuntu-badge]][Ubuntu-url]
[![IDE][vscode-badge]][vscode-url]
[![Python][Python-badge]][Python-url]

<b>
  Guilherme Alvarenga de Azevedo<br>
  Maria Eduarda Teixeira Souza<br>
</b>
  
<br>
Inteligência Computacional <br>
Engenharia de Computação <br>
CEFET-MG Campus V <br>
2026/1 


</div>

## Introdução

O presente projeto foi desenvolvido como parte do "Trabalho Intermediário - Agrupamento e Sistemas Fuzzy" da disciplina de Inteligência Computacional. O foco principal deste trabalho é aplicar as técnicas de agrupamento e sistemas fuzzy em conjunto.

O cenário de aplicação escolhido para este estudo de caso se baseia em uma base de dados sintética formulada pelo professor da disciplina. 

## Base de Dados

Tipo de Tarefa: A atividade propõe uma tarefa de multi-classificação. 

## 📚 O Projeto

Neste repositório você encontrará o código fonte do projeto, bem como os dados utilizados para a análise. O projeto foi desenvolvido em Python.

De uma forma compacta e organizada, os arquivos e diretórios estão dispostos da seguinte forma:

  ```.
clustering-fuzzy-system-ic/ 
    ├── datasets/
    │   └── base_sintetica_media.csv
    ├── src/
    │   ├── data/
    │   │   ├── loader.py   
    │   │   └── preprocessing.py            
    │   ├── eda/
    │   │   └── eda.py               
    │   ├── models/
    │   │   ├── fuzzy_cmeans.py            
    │   │   └── takagi_sugeno.py
    │   ├── utils/
    │   │   ├── metrics.py
    │   │   └── plots.py
    │   ├── config.py
    │   ├── experiments.py
    │   └── main.py
    ├── .gitignore
    ├── README.md
    └── requirements.txt
  ```

## Instalando
Para instalar o projeto, siga os passos abaixo:

<div align="justify">
  Com o ambiente preparado, os seguintes passos são para a instalação, compilação e execução do programa localmente:

  1. Clone o repositório no diretório desejado:
  ```console
  git clone https://github.com/alvarengazv/clustering-fuzzy-system-ic.git
  cd clustering-fuzzy-system-ic
  ```
  2. Crie e ative um ambiente virtual (recomendado) - garanta que já possui o [Python](https://www.python.org/downloads/), no mínimo na versão 3.11.9:
  ```console
  python3 -m venv venv
  source venv/bin/activate   # Linux/macOS
  venv\Scripts\activate      # Windows
  ```
  3. Instale as dependências com pip: 
  ```console
    pip install -r requirements.txt
  ```
</div>
<div align="justify">
  
  4. Execute o programa:
      - **Linux/macOS**
        ```console
          # Usando Python diretamente
          # PYTHONPATH='src' python3 -m main
        ```

      - **Windows**
        ```console
          # Usando Python diretamente
          # python3 src/main.py
        ```
</div> 

<div align="justify">
  
  ## Dependências

  O projeto utiliza as seguintes bibliotecas:

  - pandas
  - numpy
  - matplotlib
  - seaborn
  - scikit-learn
  - scikit-fuzzy

</div>

> [!NOTE]
> No arquivo [`requirements.txt`](requirements.txt) tem todas essas informações.

<p align="right">(<a href="#readme-topo">voltar ao topo</a>)</p>

## 🧪 Ambiente de Compilação e Execução

<div align="justify">

  O trabalho foi desenvolvido e testado em várias configurações de hardware. Podemos destacar algumas configurações de Sistema Operacional e Compilador, pois as demais configurações não influenciam diretamente no desempenho do programa.

</div>

<div align='center'>

[![SO][Ubuntu-badge]][Ubuntu-url]
[![IDE][vscode-badge]][vscode-url]
[![Python][Python-badge]][Python-url]

| *Hardware* | *Especificações* |
|:------------:|:-------------------:|
| *Laptop*   | Dell Inspiron 13 5330 |
| *Processador* | Intel Core i7-1360P |
| *Memória RAM* | 16 GB DDR5 |
| *Sistema Operacional* | Ubuntu 24.04 |
| *IDE* | Visual Studio Code |
| *Placa de Vídeo* | Intel Iris Xe Graphics |

</div>

> [!IMPORTANT] 
> Para que os testes tenham validade, considere as especificações
> do ambiente de compilação e execução do programa.

<p align="right">(<a href="#readme-topo">voltar ao topo</a>)</p>

## 📨 Contato

<div align="center">
  <br><br>
     <i>Guilherme Alvarenga de Azevedo - Graduando - 7º Período de Engenharia de Computação @ CEFET-MG</i>
  <br><br>
  
  [![Gmail][gmail-badge]][gmail-autor1]
  [![Linkedin][linkedin-badge]][linkedin-autor1]
  [![Telegram][telegram-badge]][telegram-autor1]
  
  
  <br><br>
     <i>Maria Eduarda Teixeira Souza - Graduando - 7º Período de Engenharia de Computação @ CEFET-MG</i>
  <br><br>
  
  [![Gmail][gmail-badge]][gmail-autor2]
  [![Linkedin][linkedin-badge]][linkedin-autor2]
  [![Telegram][telegram-badge]][telegram-autor2]

</div>

<p align="right">(<a href="#readme-topo">voltar ao topo</a>)</p>

<a name="referencias">📚 Referências</a>

1. AZEVEDO, Guilherme A. SOUZA, Maria E. T. **CLUSTERING-FUZZY-SYSTEM-IC**: Atividade Prática 01 - Metodologia Experimental. 2026. Disponível em: [https://github.com/alvarengazv/clustering-fuzzy-system-ic](https://github.com/alvarengazv/clustering-fuzzy-system-ic) Acesso em: 31 mar. 2026.

2. Kaggle. **Financial Fraud Detection Dataset — 1M Transactions**. Disponível em: [https://www.kaggle.com/datasets/sergionefedov/fraud-detection-1m-transactions-7-fraud-types](https://www.kaggle.com/datasets/sergionefedov/fraud-detection-1m-transactions-7-fraud-types) Acesso em: 31 mar. 2026.

3. SILVA, Alisson M. **Inteligência Computacional**: Dados. Slides de Aula. 2026.

4. SILVA, Alisson M. **Inteligência Computacional**: Metodologia dos Experimentos. Slides de Aula. 2026.

5. SILVA, Alisson M. **Inteligência Computacional**: Notas de Aula 05 - Fuzzy Introdução. 2026.
6. SILVA, Alisson M. **Inteligência Computacional**: Notas de Aula 06 - Fuzzy Conjuntos. 2026.
7. SILVA, Alisson M. **Inteligência Computacional**: Notas de Aula 07 - Fuzzy Operações. 2026.
8. SILVA, Alisson M. **Inteligência Computacional**: Notas de Aula 08 - Fuzzy Inferência 01. 2026.
9. SILVA, Alisson M. **Inteligência Computacional**: Notas de Aula 09 - Fuzzy Inferência 02. 2026.
10. SILVA, Alisson M. **Inteligência Computacional**: Notas de Aula 10 - Agrupamento. 2026.


[vscode-badge]: https://img.shields.io/badge/Visual%20Studio%20Code-0078d7.svg?style=for-the-badge&logo=visual-studio-code&logoColor=white
[vscode-url]: https://code.visualstudio.com/docs/?dv=linux64_deb
[make-badge]: https://img.shields.io/badge/_-MAKEFILE-427819.svg?style=for-the-badge
[make-url]: https://www.gnu.org/software/make/manual/make.html
[cpp-badge]: https://img.shields.io/badge/c++-%2300599C.svg?style=for-the-badge&logo=c%2B%2B&logoColor=white
[cpp-url]: https://en.cppreference.com/w/cpp
[trabalho-url]: https://drive.google.com/file/d/1-IHbGaA1BIC6_CMBydOC-NbV2bCERc8r/view?usp=sharing
[github-prof]: https://github.com/mpiress
[main-ref]: src/main.cpp
[branchAMM-url]: https://github.com/alvarengazv/trabalhosAEDS1/tree/AlgoritmosMinMax
[makefile]: ./makefile
[bash-url]: https://www.hostgator.com.br/blog/o-que-e-bash/
[lenovo-badge]: https://img.shields.io/badge/lenovo%20laptop-E2231A?style=for-the-badge&logo=lenovo&logoColor=white
[ubuntu-badge]: https://img.shields.io/badge/Ubuntu-E95420?style=for-the-badge&logo=ubuntu&logoColor=white
[Ubuntu-url]: https://ubuntu.com/
[ryzen5500-badge]: https://img.shields.io/badge/AMD%20Ryzen_5_5500U-ED1C24?style=for-the-badge&logo=amd&logoColor=white
[ryzen3500-badge]: https://img.shields.io/badge/AMD%20Ryzen_5_3500X-ED1C24?style=for-the-badge&logo=amd&logoColor=white
[windows-badge]: https://img.shields.io/badge/Windows-0078D6?style=for-the-badge&logo=windows&logoColor=white
[gcc-badge]: https://img.shields.io/badge/GCC-5C6EB8?style=for-the-badge&logo=gnu&logoColor=white
[Python-badge]: https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white
[Python-url]: https://www.python.org/


[linkedin-autor1]: https://www.linkedin.com/in/guilherme-alvarenga-de-azevedo-959474201/
[telegram-autor1]: https://t.me/alvarengazv
[gmail-autor1]: mailto:gui.alvarengas234@gmail.com

[linkedin-autor2]: https://www.linkedin.com/in/dudatsouza/
[telegram-autor2]: https://t.me/dudat_18
[gmail-autor2]: mailto:dudateixeirasouza@gmail.com

[linkedin-badge]: https://img.shields.io/badge/-LinkedIn-0077B5?style=for-the-badge&logo=Linkedin&logoColor=white
[telegram-badge]: https://img.shields.io/badge/Telegram-2CA5E0?style=for-the-badge&logo=telegram&logoColor=white
[gmail-badge]: https://img.shields.io/badge/-Gmail-D14836?style=for-the-badge&logo=Gmail&logoColor=white
