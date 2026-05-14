<div align="center">

[![Instituto Infnet](https://img.shields.io/badge/Instituto-Infnet-red?style=for-the-badge)](https://www.infnet.edu.br)
[![Curso](https://img.shields.io/badge/Curso-Engenharia_de_Software-blue?style=for-the-badge)](https://www.infnet.edu.br)
[![Bloco](https://img.shields.io/badge/Bloco_05-IA_%26_Machine_Learning-green?style=for-the-badge)](https://www.infnet.edu.br)

[![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)](https://python.org)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-latest-orange?logo=scikit-learn)](https://scikit-learn.org)
[![Pandas](https://img.shields.io/badge/Pandas-latest-150458?logo=pandas)](https://pandas.pydata.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-latest-red)](https://xgboost.readthedocs.io)
[![SHAP](https://img.shields.io/badge/SHAP-latest-purple)](https://shap.readthedocs.io)
[![NLTK](https://img.shields.io/badge/NLTK-latest-green)](https://www.nltk.org)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebooks-orange?logo=jupyter)](https://jupyter.org)
[![Status](https://img.shields.io/badge/Status-Completo-success)](https://github.com/andrebecker84/ml-portfolio)

# ml-portfolio — Machine Learning Projects

> **22 notebooks desenvolvidos ao longo do Bloco 05 (IA & Machine Learning) do curso de Engenharia de Software no Instituto Infnet — cobrindo classificação supervisionada, aprendizado não supervisionado, feature engineering, NLP e interpretabilidade de modelos.**

[![LinkedIn](https://img.shields.io/badge/LinkedIn-@becker84-0077B5?logo=linkedin)](https://linkedin.com/in/becker84)
[![GitHub](https://img.shields.io/badge/GitHub-@andrebecker84-181717?logo=github&logoColor=white)](https://github.com/andrebecker84)

</div>

---

## Índice

- [Estrutura do Repositório](#estrutura-do-repositório)
- [Notebooks](#notebooks)
- [Tópicos Cobertos](#tópicos-cobertos)
- [Stack](#stack)
- [Referências](#referências)

---

## Estrutura do Repositório

```
ml-portfolio/
├── 01_classification_basics/
│   ├── knn_logistic_regression.ipynb            # DR1-AT: KNN, Logistic Regression, GridSearchCV, ROC/AUC
│   ├── decision_tree_random_forest.ipynb        # DR1-TP3: Decision Tree, Random Forest, teste t pareado
│   └── knn_model_lifecycle.ipynb                # DR1-TP2: KNN, LinearRegression, ciclo de vida do modelo
├── 02_feature_engineering/
│   ├── feature_engineering_nlp_preprocessing.ipynb  # DR2-AT: Scaling, encoding, PCA, TF-IDF, NLTK
│   ├── text_classification_tfidf.ipynb              # DR2-TP3 (p1): Pipeline TF-IDF + Logistic Regression
│   ├── feature_normalization_selection.ipynb         # DR2-TP2 (p1-2): Normalização, RFE, Lasso
│   ├── nlp_text_preprocessing.ipynb                 # DR2-TP2 (p3): BoW, n-grams, stemming, lemmatização
│   └── categorical_encoding.ipynb                   # DR2-TP3 (p2): One-Hot, Dummy, Effect, Hashing
├── 03_advanced_classifiers/
│   ├── xgboost_svm_shap_lime.ipynb              # DR3-AT: XGBoost, SVM, GridSearchCV, SHAP, LIME
│   ├── decision_tree_titanic.ipynb              # DR3-TP1: Decision Tree, Titanic, análise SHAP
│   ├── svm_kernels_titanic.ipynb                # DR3-TP2: SVM kernels linear/RBF/Sigmoid
│   └── random_forest_xgboost_customers.ipynb   # DR3-TP3: Random Forest + XGBoost comparação
├── 04_unsupervised_learning/
│   ├── kmeans_dbscan_pca_tsne_nmf_lda.ipynb     # DR4-AT: KMeans, DBSCAN, PCA, t-SNE, NMF, LDA
│   ├── hierarchical_clustering_dendrogram.ipynb # DR4-TP2: AgglomerativeClustering, dendrogram scipy
│   ├── kmeans_pca_shopmania.ipynb               # DR4-TP1: KMeans + PCA (ShopMania 313k produtos)
│   └── dbscan_clustering.ipynb                  # DR4-TP3: DBSCAN, KMeans, Agglomerative comparação
├── 05_nlp/
│   └── sentiment_analysis_tfidf_logistic.ipynb  # Análise de sentimentos, TF-IDF, classificação
└── 06_projeto_de_bloco/                         # Série completa — Projeto de Bloco IA & ML
    ├── 01_knn_classification.ipynb              # PB-TP1: KNN baseline
    ├── 02_svm_classification.ipynb              # PB-TP2: SVM com múltiplos kernels
    ├── 03_random_forest_xgboost.ipynb           # PB-TP3: Random Forest + XGBoost
    ├── 04_kmeans_feature_engineering.ipynb      # PB-TP4: KMeans → features → classificação
    └── 05_text_classification_tfidf.ipynb       # PB-TP5: TF-IDF + classificação de texto
```

---

## Notebooks

### 01 · Classification Basics

| Notebook | Descrição |
|---|---|
| `knn_logistic_regression.ipynb` | KNN e Logistic Regression com GridSearchCV, ROC/AUC e métricas completas |
| `decision_tree_random_forest.ipynb` | Decision Tree e Random Forest com feature importance e teste t pareado |
| `knn_model_lifecycle.ipynb` | KNN e LinearRegression — ciclo de vida do modelo, treino, avaliação e persistência |

### 02 · Feature Engineering

| Notebook | Descrição |
|---|---|
| `feature_engineering_nlp_preprocessing.ipynb` | Scaling, encoding, PCA, TruncatedSVD, TF-IDF e NLTK (pipeline completo) |
| `text_classification_tfidf.ipynb` | Pipeline TF-IDF + Logistic Regression para classificação de texto |
| `feature_normalization_selection.ipynb` | Normalização (StandardScaler, L1/L2), seleção de features (filtro/Pearson, RFE, Lasso) |
| `nlp_text_preprocessing.ipynb` | Pré-processamento NLP: BoW, n-grams, stopwords NLTK, stemming (RSLP), lemmatização (spaCy) |
| `categorical_encoding.ipynb` | Encoding categórico: One-Hot, Dummy, Effect (Sum), Feature Hashing, Bin Counting |

### 03 · Advanced Classifiers

| Notebook | Descrição | Colab |
|---|---|---|
| `xgboost_svm_shap_lime.ipynb` | XGBoost + SVM + GridSearchCV + SHAP + LIME (dataset Diamonds) | — |
| `decision_tree_titanic.ipynb` | Decision Tree no Titanic — max_depth experiments e análise SHAP | — |
| `svm_kernels_titanic.ipynb` | SVM com kernels linear, RBF e Sigmoid no Titanic | — |
| `random_forest_xgboost_customers.ipynb` | Random Forest vs XGBoost — comparação em customers dataset | — |

### 04 · Unsupervised Learning

| Notebook | Descrição |
|---|---|
| `kmeans_dbscan_pca_tsne_nmf_lda.ipynb` | Pipeline completo: KMeans, DBSCAN, PCA, t-SNE, NMF, LDA (Mall Customers, MNIST, 20newsgroups) |
| `hierarchical_clustering_dendrogram.ipynb` | Clustering hierárquico — Single/Ward/Complete linkage e dendrogramas (Swiss Roll) |
| `kmeans_pca_shopmania.ipynb` | KMeans + PCA/TruncatedSVD em catálogo real ShopMania (313.706 produtos, TF-IDF) |
| `dbscan_clustering.ipynb` | DBSCAN + KMeans + Agglomerative — comparação ARI/NMI/Silhouette (Olivetti Faces, Covertypes) |

### 05 · NLP

| Notebook | Descrição |
|---|---|
| `sentiment_analysis_tfidf_logistic.ipynb` | Análise de sentimentos — pipeline TF-IDF + Logistic Regression com métricas completas |

### 06 · Projeto de Bloco — Série Completa

Série progressiva desenvolvida ao longo do Projeto de Bloco, cobrindo classificação supervisionada, SVM, ensemble methods, feature engineering não supervisionada e NLP.

<div align="center">

[![Prévia da apresentação](https://drive.google.com/thumbnail?id=1EG_clsJNdOgSpa61jL4ieBnXsNTmyVp0&sz=w640)](https://drive.google.com/file/d/1EG_clsJNdOgSpa61jL4ieBnXsNTmyVp0/view?usp=sharing)

[![Assistir apresentação](https://img.shields.io/badge/▶_Apresentação_do_Projeto_de_Bloco-Google_Drive-4285F4?style=for-the-badge&logo=googledrive&logoColor=white)](https://drive.google.com/file/d/1EG_clsJNdOgSpa61jL4ieBnXsNTmyVp0/view?usp=sharing)

</div>

| Notebook | Tema | Colab |
|---|---|---|
| `01_knn_classification.ipynb` | KNN baseline + avaliação | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1eD82WIQ87UoW0CMIMV6v3rwLIuL1Ubey?usp=sharing) |
| `02_svm_classification.ipynb` | SVM com múltiplos kernels | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1eONXTZzERp4ebtz5SyFfqoBGd9GrG56B?usp=sharing) |
| `03_random_forest_xgboost.ipynb` | Random Forest + XGBoost | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1kmHzfgL3oHGi5fA-Ife-Z-vFqnirvx7w?usp=sharing) |
| `04_kmeans_feature_engineering.ipynb` | KMeans → features → SVM + RF | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1VaYTUzoAGHcNYdaw5b5cUfdF7fJBk0uk?usp=sharing) |
| `05_text_classification_tfidf.ipynb` | TF-IDF + classificação de texto | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1ZlICpZTe42V_46hTiooiaIT-tzOCdGAq?usp=sharing) |

> Os notebooks sem badge Colab estão disponíveis localmente neste repositório.

---

## Tópicos Cobertos

### 01 · Classification Basics
- K-Nearest Neighbors (KNN) com otimização de hiperparâmetros via GridSearchCV
- Ciclo de vida do modelo: treino, avaliação, persistência e predição em novos dados
- Logistic Regression com regularização L1/L2
- Decision Trees e Random Forests com análise de feature importance
- Classe `GerenciadorExperimentos` com holdout, cross-validation e teste t pareado
- Métricas: confusion matrix, ROC/AUC, precision, recall, F1

### 02 · Feature Engineering
- Normalização numérica: StandardScaler, L1/L2 Normalizer
- Seleção de features: filtro estatístico (Pearson), RFE, Lasso/Embedding
- Scaling avançado: PowerTransformer, KBinsDiscretizer
- Encoding categórico: OneHotEncoder, DummyEncoder, Effect (Sum) Coding, Feature Hashing, Bin Counting
- Pré-processamento NLP: tokenização NLTK, stopwords, stemming (RSLPStemmer), lemmatização (spaCy)
- Representações de texto: Bag of Words, n-grams, TF-IDF vectorization
- Redução de dimensionalidade: PCA, TruncatedSVD

### 03 · Advanced Classifiers
- Decision Tree com experimentação de max_depth e análise SHAP
- Support Vector Machines com kernels linear, RBF e Sigmoid
- Random Forest vs XGBoost — comparação em dataset de clientes
- XGBoost com GridSearchCV multiclasse (dataset Diamonds)
- Interpretabilidade: SHAP values globais e LIME explanations locais

### 04 · Unsupervised Learning
- Clustering: KMeans, MiniBatchKMeans, AgglomerativeClustering, DBSCAN
- Clustering hierárquico: linkage Single/Ward/Complete, dendrogramas e heatmaps (Swiss Roll)
- Clustering em larga escala: ShopMania (313.706 produtos) com KMeans + TF-IDF + PCA/TruncatedSVD
- Comparação de algoritmos: KMeans vs DBSCAN vs Agglomerative (ARI, NMI, Silhouette)
- Redução de dimensionalidade: PCA, t-SNE, NMF, LDA
- Aplicações: imagens (Olivetti Faces), texto (20 newsgroups), consumo (Mall Customers), MNIST

### 05 · NLP
- Pipeline de análise de sentimentos (TF-IDF + Logistic Regression)
- Pré-processamento com NLTK
- Classificação binária com métricas completas

### 06 · Projeto de Bloco — Série Completa
- **TP1:** KNN com avaliação de hiperparâmetros (baseline do projeto)
- **TP2:** SVM com kernels linear, RBF e polinomial
- **TP3:** Random Forest + XGBoost com comparação de ensemble methods
- **TP4:** K-Means como feature engineering alimentando classificadores supervisionados
- **TP5:** Classificação de texto com TF-IDF, análise de silhouette e métricas NLP

---

## Stack

| Biblioteca | Finalidade |
|---|---|
| scikit-learn | Algoritmos ML, pipelines, avaliação |
| pandas | Manipulação de dados |
| numpy | Computação numérica |
| matplotlib / seaborn | Visualização |
| xgboost | Gradient boosting |
| shap | Interpretabilidade global |
| lime | Explicações locais |
| nltk | Pré-processamento NLP |
| spacy | Lemmatização |
| scipy | Clustering hierárquico |

---

## Referências

- GÉRON, Aurélien. *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow*. 3. ed. O'Reilly Media, 2022.
- RASCHKA, Sebastian; MIRJALILI, Vahid. *Python Machine Learning*. 3. ed. Packt Publishing, 2019.
- scikit-learn. *Scikit-learn Documentation*. 2024.
- NLTK Project. *Natural Language Processing with Python*. 2024.

---

<div align="center">

<p><strong>Desenvolvido como parte do Bloco 05 — Inteligência Artificial e Machine Learning.</strong></p>

<p>
  <a href="https://python.org/"><img src="https://img.shields.io/badge/Made%20with-Python-blue?logo=python" alt="Python"></a>
  <a href="https://scikit-learn.org/"><img src="https://img.shields.io/badge/Built%20with-Scikit--learn-orange?logo=scikit-learn" alt="Scikit-learn"></a>
  <a href="https://jupyter.org/"><img src="https://img.shields.io/badge/Powered%20by-Jupyter-orange?logo=jupyter" alt="Jupyter"></a>
</p>

<a href="doc/images/card.svg">
  <img src="doc/images/card.svg" width="360" alt="Becker - Software Engineer">
</a>

<p><em>Instituto Infnet — Engenharia de Software — 2025–2026.</em></p>

</div>
