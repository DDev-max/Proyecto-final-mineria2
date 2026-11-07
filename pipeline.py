import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

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
        print(self._df.describe(include="str"))

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
                .str.startswith("C", na=False) # <= Si el codigo de la factura empieza con "C", significa
            ]                                  # que la compra fue cancelada

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





