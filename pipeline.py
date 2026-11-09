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
        # Cambiado applymap por map para compatibilidad con Pandas moderno
        self._basket_sets = basket.map(lambda x: 1 if x >= 1 else 0)
        print(f"Matriz lista: {self._basket_sets.shape[0]} transacciones, {self._basket_sets.shape[1]} productos.")

    def apriori(self, min_soporte=0.01, max_len=None):
        start = time.time()
        # Usamos el prefijo explícito o aseguramos que mlxtend.frequent_patterns.apriori no choque
        from mlxtend.frequent_patterns import apriori as _apriori
        self._itemsets_apriori = _apriori(self._basket_sets, min_support=min_soporte, max_len=max_len,
                                          use_colnames=True)
        self._tiempo_apriori = time.time() - start
        print(f"Apriori completado en {self._tiempo_apriori:.4f}s. Itemsets: {len(self._itemsets_apriori)}")
        return self._itemsets_apriori

    def fp_growth(self, min_soporte=0.01, max_len=None):
        start = time.time()
        self._itemsets_fpgrowth = fpgrowth(self._basket_sets, min_support=min_soporte, max_len=max_len, use_colnames=True)
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
