import pandas as pd
import os

class PipelineEDA:
    # Se mantienen privadas las propiedades y se crean los get para evitar que sean modificadas
    def __init__(self, ruta_archivo):
        self._ruta_archivo = ruta_archivo
        self._nombre_archivo = os.path.basename(ruta_archivo)
        self._df = pd.read_csv(ruta_archivo)

    @property
    def ruta_archivo(self):
        return self._ruta_archivo

    @property
    def nombre_archivo(self):
        return self._nombre_archivo

    @property
    def df(self):
        return self._df.copy()

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


    def tratar_nulos(self, columnas, metodo="eliminar"):

        for columna in columnas:
            if metodo == "eliminar":
                self._df = self._df.dropna(subset=columnas)
                print(f"Se eliminaron los nulos de: {columna}")


    def eliminar_columnas(self, columnas):
        self._df = self._df.drop(columns=columnas)
        print(f"Columnas eliminadas: {columnas}")



    def analizar(self):
        self.informacion()
        self.ver_nulos()
