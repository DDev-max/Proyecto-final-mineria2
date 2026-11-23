import numpy as np
import plotly.express as px
from math import pi
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
import math
from collections import Counter, defaultdict
from itertools import combinations
from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules
from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules
from mlxtend.preprocessing import TransactionEncoder


class PipelineEDA:
    # Se mantienen privadas las propiedades y se crean los get para evitar que sean modificadas
    def __init__(self, ruta_archivo):
        self._ruta_archivo = ruta_archivo
        self._nombre_archivo = os.path.basename(ruta_archivo)
        self._df = pd.read_csv(ruta_archivo)
        self._filas_iniciales = len(self._df)

        # Estado de la etapa de clusterización
        self._semilla = 42
        self._rfm = None  # RFM en unidades originales
        self._datos_est = None  # RFM transformado y estandarizado
        self._proyecciones = {}  # {metodo: DataFrame dim1, dim2}
        self._clusters = {}  # {metodo: etiquetas}
        self._k = None

    @property
    def ruta_archivo(self):
        return self._ruta_archivo

    @property
    def nombre_archivo(self):
        return self._nombre_archivo

    @property
    def df(self):
        return self._df.copy()

    # Imprime cuántas filas se fueron en cada paso
    def _reporte(self, filas_antes):
        print(f"Filas eliminadas: {filas_antes - len(self._df):,}")
        print(f"Filas restantes: {len(self._df):,}")

    # ----------------- ANALISIS -----------------

    def informacion(self):
        print("=" * 50)
        print("ARCHIVO:", self.nombre_archivo)
        print("=" * 50)

        print("\nINFORMACION GENERAL DEL DATASET:")
        self._df.info()

        print("\n")
        print(self._df.describe())

        print("\n")
        print(self._df.describe(include="object"))

        print("\nDIMENSIONES DEL DATASET:")
        print(f"Filas: {self._df.shape[0]}")
        print(f"Columnas: {self._df.shape[1]}")

    def ver_nulos(self):
        print("\n" + "=" * 50)
        print("ANALISIS DE VALORES NULOS")
        print("=" * 50)

        tabla_nulos = pd.DataFrame({
            "Cantidad de nulos": self._df.isnull().sum(),
            "Porcentaje de nulos": self._df.isnull().mean() * 100
        })

        print("\nTABLA DE VALORES NULOS:")
        print(tabla_nulos)

        total_nulos = self._df.isnull().sum().sum()

        print("\nTOTAL DE VALORES NULOS:")
        print(total_nulos)

    def ver_duplicados(self):
        print(f"Filas duplicadas: {self._df.duplicated().sum():,}")

    def analizar(self):
        self.informacion()
        self.ver_nulos()

    # ----------------- LIMPIEZA -----------------

    def tratar_nulos(self, columnas, metodo="eliminar"):

        for columna in columnas:
            if metodo == "eliminar":
                self._df = self._df.dropna(subset=columnas)
                print(f"Se eliminaron los nulos de: {columna}")

    def eliminar_duplicados(self):
        filas_antes = len(self._df)
        self._df = self._df.drop_duplicates()
        self._reporte(filas_antes)

    def convertir_tipos(self, columnas_int=None, columnas_fecha=None):
        columnas_int = columnas_int or []
        columnas_fecha = columnas_fecha or []

        for columna in columnas_int:
            self._df[columna] = self._df[columna].astype("Int64")

        for columna in columnas_fecha:
            self._df[columna] = pd.to_datetime(
                self._df[columna],
                errors="coerce"
            )

    def filtrar_facturas_canceladas(self, obtener=False):
        if obtener:
            filas_canceladas = self._df[
                self._df["InvoiceNo"]
                .astype("string")
                .str.startswith("C", na=False)  # <= Si el codigo de la factura empieza con "C", significa
            ]  # que la compra fue cancelada

            return filas_canceladas.copy()

        self._df = self._df[
            ~self._df["InvoiceNo"]
            .astype("string")
            .str.startswith("C", na=False)
        ]

        print("\nSe eliminaron las facturas canceladas.")

    def filtrar_ventas_positivas(self):
        filas_antes = len(self._df)

        filtro = (self._df["Quantity"] > 0) & (self._df["UnitPrice"] > 0)
        self._df = self._df.loc[filtro].copy()
        self._reporte(filas_antes)

    def eliminar_columnas(self, columnas):
        self._df = self._df.drop(columns=columnas)
        print(f"Columnas eliminadas: {columnas}")

    def filtrar_codigos_no_producto(self, codigos, obtener=False):
        # POST = envio, M = manual, D = descuento, etc. No son productos reales
        no_producto = self._df["StockCode"].isin(codigos)

        if obtener:
            return self._df[no_producto].copy()

        filas_antes = len(self._df)
        self._df = self._df[~no_producto]
        self._reporte(filas_antes)

    def limpiar_descripciones(self):
        original = self._df["Description"]
        limpia = original.str.strip().str.replace(r"\s+", " ", regex=True)

        corregidas = original.ne(limpia).sum()
        self._df["Description"] = limpia

        print(f"Filas corregidas: {corregidas:,}")

    def convertir_tipos(self, columnas_int=None, columnas_fecha=None):
        columnas_int = columnas_int or []
        columnas_fecha = columnas_fecha or []

        for columna in columnas_int:
            self._df[columna] = self._df[columna].astype("Int64")

        for columna in columnas_fecha:
            self._df[columna] = pd.to_datetime(self._df[columna], errors="coerce")

        print(self._df.dtypes)

    def eliminar_outliers(self, columnas, percentil=0.99):
        # Quita el % mas extremo de cada columna
        filas_antes = len(self._df)

        for columna in columnas:
            limite = self._df[columna].quantile(percentil)
            self._df = self._df[self._df[columna] <= limite]
            print(f"{columna}: limite = {limite:,.2f}")

        self._reporte(filas_antes)

    def crear_columnas(self):
        self._df["Total"] = self._df["Quantity"] * self._df["UnitPrice"]
        self._df["Mes"] = self._df["InvoiceDate"].dt.to_period("M")
        self._df["Dia"] = self._df["InvoiceDate"].dt.day_name()
        self._df["Hora"] = self._df["InvoiceDate"].dt.hour

        print("Columnas nuevas: Total, Mes, Dia, Hora")

    def eliminar_columnas(self, columnas):
        self._df = self._df.drop(columns=columnas)
        print(f"Columnas eliminadas: {columnas}")

    def resumen_limpieza(self):
        eliminadas = self._filas_iniciales - len(self._df)
        print("=" * 50)
        print("RESUMEN DE LIMPIEZA")
        print("=" * 50)
        print(f"Filas iniciales: {self._filas_iniciales:,}")
        print(f"Filas finales: {len(self._df):,}")
        print(f"Eliminadas: {eliminadas:,} ({eliminadas / self._filas_iniciales:.1%})")
        print(f"Nulos restantes: {self._df.isnull().sum().sum()}")
        print(f"Duplicados restantes: {self._df.duplicated().sum()}")

    def guardar(self, ruta_salida):
        self._df.to_csv(ruta_salida, index=False)
        print(f"Guardado en: {ruta_salida}")

        # ----------------- EDA -----------------

    def resumen_general(self):
        print("=" * 50)
        print("RESUMEN GENERAL")
        print("=" * 50)
        print(f"Facturas : {self._df['InvoiceNo'].nunique():,}")
        print(f"Clientes : {self._df['CustomerID'].nunique():,}")
        print(f"Productos: {self._df['StockCode'].nunique():,}")
        print(f"Países   : {self._df['Country'].nunique():,}")
        print(f"Ventas   : {self._df['Total'].sum():,.2f}")

    def graficar_distribuciones(self):
        self._df["Quantity"].plot(kind="hist", bins=50, title="Quantity")
        plt.show()

        self._df["UnitPrice"].plot(kind="hist", bins=50, title="UnitPrice")
        plt.show()

    def ventas_por_pais(self):
        ventas = self._df.groupby("Country")["Total"].sum().sort_values()

        ventas.tail(10).plot(kind="barh", title="Top 10 países")
        plt.show()

        # Reino Unido domina todo, asi que lo quitamos para ver el resto
        ventas.drop("United Kingdom").tail(10).plot(kind="barh", title="Top 10 países (sin Reino Unido)")
        plt.show()

    def ventas_en_el_tiempo(self):
        self._df.groupby("Mes")["Total"].sum().plot(marker="o", title="Ventas por mes")
        plt.show()

        dias = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        self._df.groupby("Dia")["Total"].sum().reindex(dias).plot(kind="bar", title="Ventas por día")
        plt.show()

        self._df.groupby("Hora")["Total"].sum().plot(kind="bar", title="Ventas por hora")
        plt.show()

    def top_productos(self, n=10):
        por_producto = self._df.groupby("Description")

        por_producto["Quantity"].sum().sort_values().tail(n).plot(kind="barh", title="Más vendidos (unidades)")
        plt.show()

        por_producto["Total"].sum().sort_values().tail(n).plot(kind="barh", title="Más vendidos (dinero)")
        plt.show()

    def top_clientes(self, n=10):
        print("\n" + "=" * 50)
        print(f"TOP {n} CLIENTES POR VENTAS")
        print("=" * 50)
        print(self._df.groupby("CustomerID")["Total"].sum().sort_values(ascending=False).head(n))

    def correlacion(self):
        sns.heatmap(
            self._df[["Quantity", "UnitPrice", "Total"]].corr(),
            annot=True, cmap="coolwarm", vmin=-1, vmax=1
        )
        plt.title("Correlación entre variables")
        plt.show()

        # Corre todo el EDA de una vez

    def eda(self):
        self.resumen_general()
        self.graficar_distribuciones()
        self.ventas_por_pais()
        self.ventas_en_el_tiempo()
        self.top_productos()
        self.top_clientes()
        self.correlacion()

    def preparar_transacciones(self):
        basket = (
            self._df.groupby(["InvoiceNo", "Description"])["Quantity"]
            .sum()
            .unstack()
            .reset_index()
            .fillna(0)
            .set_index("InvoiceNo")
        )
        # Matriz booleana (1 byte por celda en vez de 8 del int64): evita MemoryError en Apriori
        self._basket_sets = (basket >= 1)
        print(f"Matriz lista: {self._basket_sets.shape[0]} transacciones, {self._basket_sets.shape[1]} productos.")

    def _basket_disperso(self):
        # Versión dispersa (sparse) de la matriz de transacciones, para Apriori y FP-Growth
        return self._basket_sets.astype(pd.SparseDtype(bool, False))

    def apriori(self, min_soporte=0.01, max_len=None):
        start = time.time()
        # Usamos el prefijo explícito o aseguramos que mlxtend.frequent_patterns.apriori no choque
        from mlxtend.frequent_patterns import apriori as _apriori
        # Matriz dispersa + low_memory: evita crear arreglos de decenas de GB con muchos productos frecuentes
        self._itemsets_apriori = _apriori(self._basket_disperso(), min_support=min_soporte, max_len=max_len,
                                          use_colnames=True, low_memory=True)
        self._tiempo_apriori = time.time() - start
        print(f"Apriori completado en {self._tiempo_apriori:.4f}s. Itemsets: {len(self._itemsets_apriori)}")
        return self._itemsets_apriori

    def fp_growth(self, min_soporte=0.01, max_len=None):
        start = time.time()
        self._itemsets_fpgrowth = fpgrowth(self._basket_disperso(), min_support=min_soporte, max_len=max_len, use_colnames=True)
        self._tiempo_fpgrowth = time.time() - start
        print(f"FP-Growth completado en {self._tiempo_fpgrowth:.4f}s. Itemsets: {len(self._itemsets_fpgrowth)}")
        return self._itemsets_fpgrowth


    def eclat(self, min_soporte=0.01, max_len=None):
        start = time.time()
        num_transacciones = len(self._basket_sets)
        min_soporte_count = min_soporte * num_transacciones

        tidsets = {}
        for col in self._basket_sets.columns:
            tids = set(self._basket_sets.index[self._basket_sets[col] == 1])
            if len(tids) >= min_soporte_count:
                tidsets[frozenset([col])] = tids

        frequent_itemsets = []

        def eclat_rec(prefix, items_tids):
            items = list(items_tids.keys())
            for i in range(len(items)):
                item1 = items[i]
                tids1 = items_tids[item1]
                new_prefix = prefix | item1
                if max_len and len(new_prefix) > max_len:
                    continue
                support = len(tids1) / num_transacciones
                frequent_itemsets.append({'support': support, 'itemsets': new_prefix})
                next_items_tids = {}
                for j in range(i + 1, len(items)):
                    item2 = items[j]
                    tids2 = items_tids[item2]
                    intersection = tids1 & tids2
                    if len(intersection) >= min_soporte_count:
                        next_items_tids[item2] = intersection
                if next_items_tids:
                    eclat_rec(new_prefix, next_items_tids)

        items_list = list(tidsets.keys())
        for i in range(len(items_list)):
            item = items_list[i]
            tids = tidsets[item]
            support = len(tids) / num_transacciones
            frequent_itemsets.append({'support': support, 'itemsets': item})
            next_items_tids = {}
            for j in range(i + 1, len(items_list)):
                item2 = items_list[j]
                tids2 = tidsets[item2]
                intersection = tids & tids2
                if len(intersection) >= min_soporte_count:
                    next_items_tids[item2] = intersection
            if next_items_tids:
                eclat_rec(item, next_items_tids)

        self._itemsets_eclat = pd.DataFrame(frequent_itemsets)
        self._tiempo_eclat = time.time() - start
        print(f"Eclat completado en {self._tiempo_eclat:.4f}s. Itemsets: {len(self._itemsets_eclat)}")
        return self._itemsets_eclat


    def comparar_algoritmos(self, min_soporte=0.01, max_len=4):
        self.apriori(min_soporte, max_len)
        self.fp_growth(min_soporte, max_len)
        self.eclat(min_soporte, max_len)
        return pd.DataFrame({
            "Algoritmo": ["Apriori", "FP-Growth", "Eclat"],
            "Tiempo (s)": [self._tiempo_apriori, self._tiempo_fpgrowth, self._tiempo_eclat],
            "Itemsets Encontrados": [len(self._itemsets_apriori), len(self._itemsets_fpgrowth), len(self._itemsets_eclat)]
        })


    def graficar_tiempos(self):
        plt.figure(figsize=(8, 5))
        sns.barplot(x=["Apriori", "FP-Growth", "Eclat"],
                    y=[self._tiempo_apriori, self._tiempo_fpgrowth, self._tiempo_eclat], palette="viridis")
        plt.title("Comparación de Tiempos de Ejecución")
        plt.ylabel("Segundos")
        plt.show()


    def graficar_itemsets(self, algoritmo="fp_growth", n=10, longitud_minima=2):
        df_itemsets = self._itemsets_apriori if algoritmo == "apriori" else (
            self._itemsets_fpgrowth if algoritmo in ["fp_growth", "fpgrowth"] else self._itemsets_eclat)
        df_filtrado = df_itemsets[df_itemsets['itemsets'].apply(lambda x: len(x) >= longitud_minima)].copy()
        df_filtrado['itemsets_str'] = df_filtrado['itemsets'].apply(lambda x: ', '.join(list(x)))
        top_n = df_filtrado.sort_values(by='support', ascending=False).head(n)
        plt.figure(figsize=(10, 6))
        sns.barplot(x='support', y='itemsets_str', data=top_n, palette="mako")
        plt.title(f"Top {n} Itemsets Frecuentes ({algoritmo.upper()})")
        plt.show()


    def generar_reglas(self, algoritmo="fp_growth", min_confianza=0.5, min_lift=1.2):
        itemsets = self._itemsets_apriori if algoritmo == "apriori" else (
            self._itemsets_fpgrowth if algoritmo in ["fp_growth", "fpgrowth"] else self._itemsets_eclat)
        self._reglas = association_rules(itemsets, metric="confidence", min_threshold=min_confianza)
        if min_lift:
            self._reglas = self._reglas[self._reglas['lift'] >= min_lift]
        return self._reglas


    def top_reglas(self, algoritmo="fp_growth", n=10, ordenar_por="lift"):
        if not hasattr(self, '_reglas'):
            self.generar_reglas(algoritmo)
        return self._reglas.sort_values(by=ordenar_por, ascending=False).head(n)[
            ['antecedents', 'consequents', 'support', 'confidence', 'lift']]


    def graficar_reglas(self, algoritmo="fp_growth"):
        if not hasattr(self, '_reglas'):
            self.generar_reglas(algoritmo)
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x="support", y="confidence", size="lift", hue="lift", data=self._reglas, palette="coolwarm",
                        sizes=(20, 200))
        plt.title("Soporte vs Confianza (Reglas de Asociación)")
        plt.show()


    def guardar_reglas(self, algoritmo="fp_growth", ruta_salida="reglas_asociacion.csv"):
        if not hasattr(self, '_reglas'):
            self.generar_reglas(algoritmo)
        df_guardar = self._reglas.copy()
        df_guardar['antecedents'] = df_guardar['antecedents'].apply(lambda x: ', '.join(list(x)))
        df_guardar['consequents'] = df_guardar['consequents'].apply(lambda x: ', '.join(list(x)))
        df_guardar.to_csv(ruta_salida, index=False)
        print(f"Reglas guardadas en: {ruta_salida}")

    # ----------------- CLUSTERING: ESTADO -----------------

    METODOS_CLUSTERING = ("original", "pca", "tsne", "umap")

    @property
    def rfm(self):
        return self._rfm.copy()

    @property
    def datos_estandarizados(self):
        return self._datos_est.copy()

    @property
    def k_clusters(self):
        return self._k

    # ----------------- CLUSTERING: PREPARACIÓN (RFM) -----------------

    def construir_rfm(self, fecha_referencia=None):
        # Recencia: días desde la última compra. Frecuencia: facturas distintas. Monetario: gasto total
        datos = self._df[["CustomerID", "InvoiceNo"]].copy()
        datos["InvoiceDate"] = pd.to_datetime(self._df["InvoiceDate"], errors="coerce")
        if "Total" in self._df.columns:
            datos["Total"] = self._df["Total"]
        else:
            datos["Total"] = self._df["Quantity"] * self._df["UnitPrice"]

        if fecha_referencia is None:
            fecha_referencia = datos["InvoiceDate"].max() + pd.Timedelta(days=1)
        fecha_referencia = pd.to_datetime(fecha_referencia)

        self._rfm = datos.groupby("CustomerID").agg(
            Recencia=("InvoiceDate", lambda f: (fecha_referencia - f.max()).days),
            Frecuencia=("InvoiceNo", "nunique"),
            Monetario=("Total", "sum"),
        )
        print(f"RFM construido para {len(self._rfm):,} clientes. Fecha de referencia: {fecha_referencia.date()}")
        return self._rfm.copy()

    def estandarizar(self):
        # Las tres variables son muy asimétricas: log1p reduce el peso de los clientes extremos
        if self._rfm is None:
            self.construir_rfm()
        transformado = np.log1p(self._rfm)
        self._datos_est = pd.DataFrame(
            StandardScaler().fit_transform(transformado),
            columns=self._rfm.columns, index=self._rfm.index
        )
        print("Datos transformados (log1p) y estandarizados (media 0, desviación 1).")
        return self._datos_est.copy()

    def graficar_distribuciones_rfm(self):
        fig, ejes = plt.subplots(2, 3, figsize=(14, 7))
        for j, col in enumerate(self._rfm.columns):
            sns.histplot(self._rfm[col], bins=40, ax=ejes[0, j], color="steelblue")
            ejes[0, j].set_title(f"{col} (original)")
            sns.histplot(self._datos_est[col], bins=40, ax=ejes[1, j], color="darkorange")
            ejes[1, j].set_title(f"{col} (log1p + estandarizado)")
        plt.tight_layout()
        plt.show()

    # ----------------- ELECCIÓN DE K -----------------

    def evaluar_k(self, k_min=2, k_max=10):
        filas = []
        for k in range(k_min, k_max + 1):
            etiquetas = KMeans(n_clusters=k, n_init=20, random_state=self._semilla).fit(self._datos_est)
            lab = etiquetas.labels_
            filas.append({
                "k": k,
                "Inercia": etiquetas.inertia_,
                "Silhouette": silhouette_score(self._datos_est, lab),
                "Davies-Bouldin": davies_bouldin_score(self._datos_est, lab),
                "Calinski-Harabasz": calinski_harabasz_score(self._datos_est, lab),
            })
        self._evaluacion_k = pd.DataFrame(filas).set_index("k")
        return self._evaluacion_k.copy()

    def graficar_evaluacion_k(self):
        ev = self._evaluacion_k
        fig, ejes = plt.subplots(1, 3, figsize=(15, 4))
        ejes[0].plot(ev.index, ev["Inercia"], marker="o")
        ejes[0].set_title("Método del codo (inercia)")
        ejes[1].plot(ev.index, ev["Silhouette"], marker="o", color="green")
        ejes[1].set_title("Coeficiente de silhouette (más alto es mejor)")
        ejes[2].plot(ev.index, ev["Davies-Bouldin"], marker="o", color="crimson")
        ejes[2].set_title("Davies-Bouldin (más bajo es mejor)")
        for ax in ejes:
            ax.set_xlabel("Número de clústeres (k)")
        plt.tight_layout()
        plt.show()

    def sugerir_k(self):
        k = int(self._evaluacion_k["Silhouette"].idxmax())
        print(f"k con mayor silhouette: {k} ({self._evaluacion_k.loc[k, 'Silhouette']:.3f})")
        return k

    # ----------------- REDUCCIÓN DE DIMENSIÓN -----------------

    def proyectar(self, metodo, **kwargs):
        # Devuelve las 2 dimensiones (dim1, dim2) para graficar, igual que en el ejemplo de clase
        X = self._datos_est
        if metodo == "pca":
            pca = PCA(n_components=2, random_state=self._semilla)
            coords = pca.fit_transform(X)
            self._varianza_pca = pca.explained_variance_ratio_
        elif metodo == "tsne":
            perplexity = kwargs.get("perplexity", 30)
            coords = TSNE(n_components=2, perplexity=perplexity, learning_rate="auto",
                          init="pca", random_state=self._semilla).fit_transform(X)
        elif metodo == "umap":
            from umap.umap_ import UMAP  # import local: solo se exige si se usa UMAP
            coords = UMAP(n_components=2, n_neighbors=kwargs.get("n_neighbors", 15),
                          min_dist=kwargs.get("min_dist", 0.1),
                          random_state=self._semilla).fit_transform(X)
        else:
            raise ValueError("metodo debe ser 'pca', 'tsne' o 'umap'")

        self._proyecciones[metodo] = pd.DataFrame(coords, columns=["dim1", "dim2"], index=X.index)
        print(f"Proyección {metodo.upper()} lista.")
        return self._proyecciones[metodo].copy()

    # ----------------- CLUSTERING -----------------

    def clusterizar(self, k, metodo="original"):
        # 'original': K-Means sobre RFM estandarizado. Otros: K-Means sobre las 2 dimensiones
        # de PCA / t-SNE / UMAP (lo que hace el ejemplo de clase)
        if metodo not in self.METODOS_CLUSTERING:
            raise ValueError(f"metodo debe ser uno de {self.METODOS_CLUSTERING}")
        if metodo == "original":
            espacio = self._datos_est
        else:
            if metodo not in self._proyecciones:
                self.proyectar(metodo)
            espacio = self._proyecciones[metodo][["dim1", "dim2"]]

        km = KMeans(n_clusters=k, max_iter=2000, n_init=50, random_state=self._semilla).fit(espacio)
        self._clusters[metodo] = km.labels_
        self._k = k
        return km.labels_

    def clusterizar_todos(self, k):
        for metodo in self.METODOS_CLUSTERING:
            self.clusterizar(k, metodo)
        return self.comparar_metodos_clustering()

    def comparar_metodos_clustering(self):
        # Todas las métricas se calculan sobre el espacio original estandarizado para que sean comparables,
        # y además sobre el espacio donde se hizo el clustering
        filas = []
        for metodo, lab in self._clusters.items():
            if metodo == "original":
                espacio_propio = self._datos_est.values
            else:
                espacio_propio = self._proyecciones[metodo][["dim1", "dim2"]].values
            filas.append({
                "Método": metodo.upper() if metodo != "original" else "K-Means directo",
                "k": len(set(lab)),
                "Silhouette (espacio RFM)": silhouette_score(self._datos_est, lab),
                "Silhouette (espacio del método)": silhouette_score(espacio_propio, lab),
                "Davies-Bouldin (RFM)": davies_bouldin_score(self._datos_est, lab),
                "Calinski-Harabasz (RFM)": calinski_harabasz_score(self._datos_est, lab),
            })
        return pd.DataFrame(filas).set_index("Método").round(3)

    # ----------------- VISUALIZACIÓN -----------------

    def graficar_proyeccion(self, metodo, metodo_clusters=None):
        # Pinta la proyección de 'metodo' con las etiquetas de 'metodo_clusters' (por defecto el mismo)
        metodo_clusters = metodo_clusters or metodo
        if metodo not in self._proyecciones:
            self.proyectar(metodo)
        df = self._proyecciones[metodo].copy()
        df["cluster"] = self._clusters[metodo_clusters].astype(str)
        df["CustomerID"] = df.index.astype(str)
        fig = px.scatter(df.sort_values("cluster"), x="dim1", y="dim2", color="cluster",
                         hover_data=["CustomerID"], opacity=0.7,
                         title=f"{metodo.upper()} - clústeres de {metodo_clusters.upper()}")
        fig.update_layout(title_x=0.5)
        fig.show()

    def graficar_proyecciones_estaticas(self, metodo_clusters="original"):
        # Versión matplotlib (se ve siempre en el HTML exportado, sin depender de plotly.js)
        etiquetas = self._clusters[metodo_clusters]
        metodos = [m for m in self._proyecciones]
        fig, ejes = plt.subplots(1, len(metodos), figsize=(6 * len(metodos), 5))
        ejes = np.atleast_1d(ejes)
        for ax, m in zip(ejes, metodos):
            p = self._proyecciones[m]
            sc = ax.scatter(p["dim1"], p["dim2"], c=etiquetas, cmap="tab10", s=8, alpha=0.7)
            ax.set_title(f"{m.upper()} (colores = clústeres de {metodo_clusters.upper()})")
        plt.tight_layout()
        plt.show()

    def perfiles_clusters(self, metodo="original"):
        # Promedio de RFM (unidades originales) por clúster + tamaño
        lab = self._clusters[metodo]
        tabla = self._rfm.assign(cluster=lab).groupby("cluster").agg(
            Clientes=("Recencia", "size"),
            Recencia=("Recencia", "mean"),
            Frecuencia=("Frecuencia", "mean"),
            Monetario=("Monetario", "mean"),
            Monetario_total=("Monetario", "sum"),
        )
        tabla["% clientes"] = tabla["Clientes"] / tabla["Clientes"].sum() * 100
        tabla["% ingresos"] = tabla["Monetario_total"] / tabla["Monetario_total"].sum() * 100
        return tabla.drop(columns="Monetario_total").round(2)

    def graficar_perfiles(self, metodo="original"):
        centros = self._rfm.assign(cluster=self._clusters[metodo]).groupby("cluster").mean()
        centros_rel = centros / centros.max()  # escala 0-1 por variable para poder compararlas
        centros_rel.plot(kind="bar", figsize=(9, 5), rot=0)
        plt.title(f"Perfil relativo de cada clúster ({metodo.upper()})")
        plt.ylabel("Promedio / máximo entre clústeres")
        plt.show()

    def graficar_radar(self, metodo="original"):
        centros = self._rfm.assign(cluster=self._clusters[metodo]).groupby("cluster").mean()
        etiquetas = list(centros.columns)
        # min-max por variable entre clústeres
        rango = (centros.max() - centros.min()).replace(0, 1)
        norm = (centros - centros.min()) / rango * 100
        angulos = [n / len(etiquetas) * 2 * pi for n in range(len(etiquetas))]
        angulos += angulos[:1]
        ax = plt.subplot(111, polar=True)
        ax.set_theta_offset(pi / 2)
        ax.set_theta_direction(-1)
        plt.xticks(angulos[:-1], etiquetas)
        plt.ylim(0, 100)
        for c in norm.index:
            vals = norm.loc[c].tolist()
            vals += vals[:1]
            ax.plot(angulos, vals, linewidth=1, label=f"Cluster {c}")
            ax.fill(angulos, vals, alpha=0.25)
        plt.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
        plt.title(f"Radar de clústeres ({metodo.upper()})")
        plt.show()

    # ----------------- INTERPRETACIÓN DE NEGOCIO -----------------

    def nombrar_segmentos(self, metodo="original"):
        # Etiqueta cada clúster comparando su RFM con la mediana global de los clientes
        perf = self.perfiles_clusters(metodo)
        med = self._rfm.median()
        nombres = {}
        for c, f in perf.iterrows():
            reciente = f["Recencia"] <= med["Recencia"]
            frecuente = f["Frecuencia"] >= med["Frecuencia"]
            gasta = f["Monetario"] >= med["Monetario"]
            if reciente and frecuente and gasta:
                nombres[c] = "Campeones (compran seguido, reciente y gastan mucho)"
            elif frecuente and gasta:
                nombres[c] = "Leales en enfriamiento (buenos clientes que dejan de comprar)"
            elif reciente and not frecuente:
                nombres[c] = "Nuevos / ocasionales recientes"
            elif not reciente and not frecuente and not gasta:
                nombres[c] = "Perdidos (baja actividad y bajo gasto)"
            else:
                nombres[c] = "En riesgo / valor medio"
        return pd.Series(nombres, name="Segmento")

    def resumen_negocio(self, metodo="original"):
        perf = self.perfiles_clusters(metodo)
        seg = self.nombrar_segmentos(metodo)
        print("=" * 60)
        print(f"RESUMEN DE SEGMENTOS ({metodo.upper()}, k={len(perf)})")
        print("=" * 60)
        for c, f in perf.iterrows():
            print(f"\nCluster {c} - {seg[c]}")
            print(f"  {int(f['Clientes']):,} clientes ({f['% clientes']:.1f}%) | {f['% ingresos']:.1f}% de los ingresos")
            print(f"  Última compra hace {f['Recencia']:.0f} días | {f['Frecuencia']:.1f} facturas | "
                  f"gasto promedio {f['Monetario']:,.2f}")
        return pd.concat([perf, seg], axis=1)

    def guardar_clusters(self, metodo="original", ruta_salida="clientes_segmentados.csv"):
        salida = self._rfm.assign(cluster=self._clusters[metodo])
        salida["segmento"] = salida["cluster"].map(self.nombrar_segmentos(metodo))
        salida.to_csv(ruta_salida)
        print(f"Clientes segmentados guardados en: {ruta_salida}")

    # ----------------- EJECUCIÓN COMPLETA -----------------

    def ejecutar_clustering(self, k=None, k_min=2, k_max=10, proyecciones=False,
                 graficar=True, ruta_salida="clientes_segmentados.csv"):
        """
        Corre todo el flujo de clusterización con K-Means directo sobre RFM ('original'):
        RFM -> estandarizar -> elegir k -> K-Means -> gráficos -> perfiles -> CSV.

        k            : número de clústeres. Si es None se usa el de mayor silhouette.
        proyecciones : True agrega además PCA, t-SNE y UMAP con su tabla comparativa.
        Devuelve el resumen de negocio (un DataFrame con una fila por clúster).
        """
        self.construir_rfm()
        self.estandarizar()

        self.evaluar_k(k_min, k_max)
        if graficar:
            self.graficar_evaluacion_k()
        if k is None:
            k = self.sugerir_k()

        self.clusterizar(k, "original")

        if proyecciones:
            comparacion = self.clusterizar_todos(k)
            print("\nComparación de métodos:")
            print(comparacion.to_string())
        else:
            self.proyectar("pca")  # solo para poder dibujar los clústeres en 2D

        if graficar:
            self.graficar_proyecciones_estaticas("original")
            self.graficar_perfiles("original")
            self.graficar_radar("original")

        resumen = self.resumen_negocio("original")
        self.guardar_clusters("original", ruta_salida)
        return resumen
