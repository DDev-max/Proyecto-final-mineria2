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

    # ----------------- REGLAS DE ASOCIACION -----------------
    # Canasta = factura (InvoiceNo). Item = producto (Description).
    # Los tres algoritmos devuelven los mismos itemsets frecuentes; solo cambia como los buscan.

    def preparar_transacciones(self, columna_transaccion="InvoiceNo", columna_item="Description",
                               pais=None, min_items=1):
        df = self._df
        if pais is not None:
            df = df[df["Country"] == pais]

        # Un producto se cuenta una sola vez por factura (el soporte es por factura, no por unidades)
        pares = df[[columna_transaccion, columna_item]].dropna().astype(str).drop_duplicates()
        canastas = pares.groupby(columna_transaccion)[columna_item].agg(frozenset)
        self._transacciones = [c for c in canastas if len(c) >= min_items]

        # Si cambian las transacciones, los resultados anteriores ya no sirven
        self._itemsets = {}
        self._reglas = {}

        tamanios = [len(c) for c in self._transacciones]
        print("=" * 50)
        print("TRANSACCIONES PARA REGLAS DE ASOCIACION")
        print("=" * 50)
        print(f"Transacciones (facturas): {len(self._transacciones):,}")
        print(f"Productos distintos     : {len(set().union(*self._transacciones)):,}")
        print(f"Items por factura       : promedio {sum(tamanios) / len(tamanios):.1f} | maximo {max(tamanios)}")

    def _codificar(self, min_soporte):
        # Deja solo los items frecuentes y los cambia por ids enteros (0 = el mas frecuente)
        if self._transacciones is None:
            raise ValueError("Primero ejecuta preparar_transacciones()")
        if not 0 < min_soporte <= 1:
            raise ValueError("min_soporte debe estar entre 0 y 1")

        n = len(self._transacciones)
        min_conteo = math.ceil(min_soporte * n - 1e-9)

        conteo = Counter(item for t in self._transacciones for item in t)
        frecuentes = sorted(
            (it for it, c in conteo.items() if c >= min_conteo),
            key=lambda it: (-conteo[it], it)
        )
        ids = {it: i for i, it in enumerate(frecuentes)}
        conteos = [conteo[it] for it in frecuentes]

        trans = []
        for t in self._transacciones:
            codificada = tuple(sorted(ids[it] for it in t if it in ids))
            if codificada:
                trans.append(codificada)

        return n, min_conteo, frecuentes, conteos, trans

    def _guardar_itemsets(self, algoritmo, resultado, nombres, n, min_soporte, max_len, tiempo):
        filas = [
            {
                "itemsets": frozenset(nombres[i] for i in ids),
                "conteo": conteo,
                "soporte": conteo / n,
                "longitud": len(ids),
            }
            for ids, conteo in resultado.items()
        ]
        tabla = pd.DataFrame(filas, columns=["itemsets", "conteo", "soporte", "longitud"])
        tabla = tabla.sort_values(["soporte", "longitud"], ascending=[False, True]).reset_index(drop=True)

        self._itemsets[algoritmo] = {
            "tabla": tabla, "n": n, "min_soporte": min_soporte, "max_len": max_len, "tiempo": tiempo
        }
        self._reglas.pop(algoritmo, None)

        print(f"[{algoritmo}] itemsets frecuentes: {len(tabla):,} | tiempo: {tiempo:.2f} s")
        return tabla.copy()

    # ----- Apriori -----

    @staticmethod
    def _apriori_candidatos(previos, k):
        # Une itemsets de tamano k-1 que comparten los primeros k-2 items y poda con la propiedad antimonotona
        previos_set = set(previos)
        grupos = defaultdict(list)
        for itemset in previos:
            grupos[itemset[:-1]].append(itemset[-1])

        candidatos = []
        for prefijo, ultimos in grupos.items():
            ultimos.sort()
            for a, b in combinations(ultimos, 2):
                candidato = prefijo + (a, b)
                if all(candidato[:j] + candidato[j + 1:] in previos_set for j in range(k - 2)):
                    candidatos.append(candidato)
        return candidatos

    def apriori(self, min_soporte=0.02, max_len=None):
        n, min_conteo, nombres, conteos, trans = self._codificar(min_soporte)
        inicio = time.perf_counter()

        nivel = {(i,): c for i, c in enumerate(conteos)}
        resultado = dict(nivel)

        k = 2
        while nivel and (max_len is None or k <= max_len):
            candidatos = self._apriori_candidatos(sorted(nivel), k)
            if not candidatos:
                break

            cuenta = dict.fromkeys(candidatos, 0)
            for t in trans:
                if len(t) < k:
                    continue
                if math.comb(len(t), k) <= len(candidatos):
                    for c in combinations(t, k):
                        if c in cuenta:
                            cuenta[c] += 1
                else:
                    ts = set(t)
                    for c in candidatos:
                        if ts.issuperset(c):
                            cuenta[c] += 1

            nivel = {c: v for c, v in cuenta.items() if v >= min_conteo}
            resultado.update(nivel)
            k += 1

        tiempo = time.perf_counter() - inicio
        return self._guardar_itemsets("apriori", resultado, nombres, n, min_soporte, max_len, tiempo)

    # ----- FP-Growth -----

    class _NodoFP:
        __slots__ = ("item", "conteo", "padre", "hijos")

        def __init__(self, item, padre):
            self.item = item
            self.conteo = 0
            self.padre = padre
            self.hijos = {}

    @staticmethod
    def _fp_arbol(bases, min_conteo):
        # bases: lista de (camino, peso). Devuelve la tabla de nodos por item y el conteo de cada item frecuente
        conteo = defaultdict(int)
        for camino, peso in bases:
            for item in camino:
                conteo[item] += peso
        conteo = {it: c for it, c in conteo.items() if c >= min_conteo}

        raiz = PipelineEDA._NodoFP(None, None)
        tabla = defaultdict(list)
        for camino, peso in bases:
            nodo = raiz
            for item in camino:
                if item not in conteo:
                    continue
                hijo = nodo.hijos.get(item)
                if hijo is None:
                    hijo = PipelineEDA._NodoFP(item, nodo)
                    nodo.hijos[item] = hijo
                    tabla[item].append(hijo)
                hijo.conteo += peso
                nodo = hijo
        return tabla, conteo

    def _fp_minar(self, tabla, conteo, sufijo, min_conteo, max_len, resultado):
        # Se empieza por el item menos frecuente (id mas alto); sus ancestros siempre tienen id menor
        for item in sorted(tabla, reverse=True):
            nuevo = sufijo + (item,)
            resultado[tuple(sorted(nuevo))] = conteo[item]

            if max_len is not None and len(nuevo) >= max_len:
                continue

            base = []
            for nodo in tabla[item]:
                camino = []
                padre = nodo.padre
                while padre.item is not None:
                    camino.append(padre.item)
                    padre = padre.padre
                if camino:
                    camino.reverse()
                    base.append((camino, nodo.conteo))

            if base:
                sub_tabla, sub_conteo = self._fp_arbol(base, min_conteo)
                if sub_tabla:
                    self._fp_minar(sub_tabla, sub_conteo, nuevo, min_conteo, max_len, resultado)

    def fp_growth(self, min_soporte=0.02, max_len=None):
        n, min_conteo, nombres, conteos, trans = self._codificar(min_soporte)
        inicio = time.perf_counter()

        tabla, conteo = self._fp_arbol([(t, 1) for t in trans], min_conteo)
        resultado = {}
        self._fp_minar(tabla, conteo, (), min_conteo, max_len, resultado)

        tiempo = time.perf_counter() - inicio
        return self._guardar_itemsets("fp_growth", resultado, nombres, n, min_soporte, max_len, tiempo)

    # ----- Eclat -----

    def _eclat_rec(self, prefijo, extensiones, min_conteo, max_len, resultado):
        # extensiones: lista de (item, tidset, conteo). El tidset es un entero usado como bitset de facturas
        for pos, (item, tids, conteo) in enumerate(extensiones):
            nuevo = prefijo + (item,)
            resultado[tuple(sorted(nuevo))] = conteo

            if max_len is not None and len(nuevo) >= max_len:
                continue

            siguientes = []
            for item2, tids2, _ in extensiones[pos + 1:]:
                interseccion = tids & tids2
                c = interseccion.bit_count()
                if c >= min_conteo:
                    siguientes.append((item2, interseccion, c))

            if siguientes:
                self._eclat_rec(nuevo, siguientes, min_conteo, max_len, resultado)

    def eclat(self, min_soporte=0.02, max_len=None):
        n, min_conteo, nombres, conteos, trans = self._codificar(min_soporte)
        inicio = time.perf_counter()

        # Formato vertical: para cada item, el conjunto de facturas donde aparece (bit i = factura i)
        tidsets = [0] * len(nombres)
        for tid, t in enumerate(trans):
            bit = 1 << tid
            for item in t:
                tidsets[item] |= bit

        # De menor a mayor soporte, para que las intersecciones se achiquen rapido
        iniciales = [(i, tidsets[i], conteos[i]) for i in reversed(range(len(nombres)))]
        resultado = {}
        self._eclat_rec((), iniciales, min_conteo, max_len, resultado)

        tiempo = time.perf_counter() - inicio
        return self._guardar_itemsets("eclat", resultado, nombres, n, min_soporte, max_len, tiempo)

    # ----- Comparacion -----

    def comparar_algoritmos(self, min_soporte=0.02, max_len=None):
        print("=" * 50)
        print(f"COMPARACION DE ALGORITMOS (min_soporte={min_soporte}, max_len={max_len})")
        print("=" * 50)

        for algoritmo in (self.apriori, self.fp_growth, self.eclat):
            algoritmo(min_soporte, max_len)

        conjuntos = {
            nombre: {(fs, c) for fs, c in zip(d["tabla"]["itemsets"], d["tabla"]["conteo"])}
            for nombre, d in self._itemsets.items()
        }
        iguales = conjuntos["apriori"] == conjuntos["fp_growth"] == conjuntos["eclat"]
        print(f"\nLos tres encuentran los mismos itemsets con los mismos conteos: {iguales}")

        return pd.DataFrame({
            "algoritmo": list(self._itemsets),
            "itemsets": [len(d["tabla"]) for d in self._itemsets.values()],
            "tiempo_s": [round(d["tiempo"], 3) for d in self._itemsets.values()],
        })

    def obtener_itemsets(self, algoritmo="fp_growth"):
        self._validar_algoritmo(algoritmo)
        return self._itemsets[algoritmo]["tabla"].copy()

    def _validar_algoritmo(self, algoritmo):
        if algoritmo not in self._itemsets:
            raise ValueError(f"Primero ejecuta {algoritmo}(). Disponibles: {list(self._itemsets)}")

    # ----- Reglas -----

    def generar_reglas(self, algoritmo="fp_growth", min_confianza=0.5, min_lift=1.0):
        self._validar_algoritmo(algoritmo)
        datos = self._itemsets[algoritmo]
        n = datos["n"]
        conteos = dict(zip(datos["tabla"]["itemsets"], datos["tabla"]["conteo"]))

        filas = []
        for itemset, conteo_ab in conteos.items():
            if len(itemset) < 2:
                continue
            for r in range(1, len(itemset)):
                for antecedente in combinations(sorted(itemset), r):
                    antecedente = frozenset(antecedente)
                    consecuente = itemset - antecedente

                    sop_a = conteos[antecedente] / n  # subconjuntos de un itemset frecuente tambien son frecuentes
                    sop_c = conteos[consecuente] / n
                    sop_ab = conteo_ab / n

                    confianza = sop_ab / sop_a
                    lift = confianza / sop_c
                    if confianza < min_confianza or lift < min_lift:
                        continue

                    filas.append({
                        "antecedentes": antecedente,
                        "consecuentes": consecuente,
                        "soporte_antecedente": sop_a,
                        "soporte_consecuente": sop_c,
                        "soporte": sop_ab,
                        "confianza": confianza,
                        "lift": lift,
                        "leverage": sop_ab - sop_a * sop_c,
                        "conviccion": (1 - sop_c) / (1 - confianza) if confianza < 1 else float("inf"),
                    })

        columnas = ["antecedentes", "consecuentes", "soporte_antecedente", "soporte_consecuente",
                    "soporte", "confianza", "lift", "leverage", "conviccion"]
        reglas = pd.DataFrame(filas, columns=columnas)
        reglas = reglas.sort_values(["lift", "confianza"], ascending=False).reset_index(drop=True)

        self._reglas[algoritmo] = reglas
        print(f"[{algoritmo}] reglas generadas: {len(reglas):,} "
              f"(confianza >= {min_confianza}, lift >= {min_lift})")
        return reglas.copy()

    def obtener_reglas(self, algoritmo="fp_growth"):
        if algoritmo not in self._reglas:
            raise ValueError(f"Primero ejecuta generar_reglas('{algoritmo}')")
        return self._reglas[algoritmo].copy()

    @staticmethod
    def _texto(conjunto):
        return " + ".join(sorted(conjunto))

    def top_reglas(self, algoritmo="fp_growth", n=10, ordenar_por="lift"):
        reglas = self.obtener_reglas(algoritmo).sort_values(ordenar_por, ascending=False).head(n)
        legible = reglas.copy()
        legible["antecedentes"] = legible["antecedentes"].map(self._texto)
        legible["consecuentes"] = legible["consecuentes"].map(self._texto)
        return legible[["antecedentes", "consecuentes", "soporte", "confianza", "lift"]].round(3).reset_index(drop=True)

    def guardar_reglas(self, algoritmo, ruta_salida):
        reglas = self.obtener_reglas(algoritmo)
        reglas["antecedentes"] = reglas["antecedentes"].map(self._texto)
        reglas["consecuentes"] = reglas["consecuentes"].map(self._texto)
        reglas.to_csv(ruta_salida, index=False)
        print(f"Guardado en: {ruta_salida}")

    # ----- Graficos -----

    def graficar_itemsets(self, algoritmo="fp_growth", n=10, longitud_minima=1):
        tabla = self.obtener_itemsets(algoritmo)
        tabla = tabla[tabla["longitud"] >= longitud_minima].head(n).copy()
        tabla["etiqueta"] = tabla["itemsets"].map(self._texto)
        tabla.sort_values("soporte").plot(
            x="etiqueta", y="soporte", kind="barh", legend=False,
            title=f"Itemsets mas frecuentes ({algoritmo})"
        )
        plt.xlabel("Soporte")
        plt.ylabel("")
        plt.tight_layout()
        plt.show()

    def graficar_reglas(self, algoritmo="fp_growth"):
        reglas = self.obtener_reglas(algoritmo)
        plt.figure(figsize=(8, 5))
        puntos = plt.scatter(reglas["soporte"], reglas["confianza"], c=reglas["lift"], cmap="viridis", alpha=0.7)
        plt.colorbar(puntos, label="Lift")
        plt.xlabel("Soporte")
        plt.ylabel("Confianza")
        plt.title(f"Reglas de asociacion ({algoritmo})")
        plt.show()

    def graficar_tiempos(self):
        if not self._itemsets:
            raise ValueError("Primero ejecuta algun algoritmo")
        tiempos = pd.Series({nombre: d["tiempo"] for nombre, d in self._itemsets.items()})
        tiempos.plot(kind="bar", title="Tiempo de ejecucion (segundos)")
        plt.xticks(rotation=0)
        plt.show()
