import warnings
from typing import Tuple

import math
import numpy as np
from numpy import typing as npt


def divisors(n: int | float) -> npt.NDArray[np.float64]:
    """
    Calcola tutti i divisori interi positivi di un numero.

    Ottimizzato tramite vettorizzazione NumPy. Se viene fornito un float
    con decimali, il valore viene troncato (floor) sollevando un warning
    per prevenire errori di precisione o arrotondamenti involontari.

    Args:
        n (int | float): Il numero di cui estrarre i divisori.

    Returns:
        npt.NDArray[np.int64]: Un array 1D contenente i divisori univoci,
        ordinati in modo crescente. Restituisce un array vuoto se n <= 0.

    Warns:
        UserWarning: Se `n` è un float con parte decimale non nulla.

    Example:
        >>> divisors(28)
        array([ 1,  2,  4,  7, 14, 28])
    """

    # Cast e validazione dell'ingresso per garantire un intero esatto
    if isinstance(n, float):
        if not n.is_integer():
            new_n = math.floor(n)
            warnings.warn(
                f"L'input {n} non è un intero. Valore troncato a {new_n} "
                f"(errore: {new_n - n:.2e})"
            )
            n = new_n
        else:
            n = int(n)

    # Gestione dei casi limite
    if n <= 0:
        return np.array([], dtype=np.int64)
    if n == 1:
        return np.array([1], dtype=np.int64)

    # Identificazione dei candidati fino alla radice quadrata
    limit = int(n**0.5)
    candidates = np.arange(1, limit + 1, dtype=np.int64)

    # Filtraggio dei divisori esatti tramite modulo
    divs = candidates[n % candidates == 0]

    # Calcolo dei complementari e rimozione dei duplicati (es. Quadrati perfetti)
    all_divs = np.unique(np.concatenate((divs, n // divs)))

    return all_divs


def common_divisors(
    ns: npt.NDArray[np.float64],
) -> Tuple[int | float, npt.NDArray[np.float64]]:
    """
    Calcola il Massimo Comune Divisore (MCD) e tutti i divisori comuni di un array.

    Il calcolo avviene estraendo prima l'MCD dell'intero array tramite riduzione,
    per poi calcolarne i divisori. Se l'array contiene float con decimali,
    i valori vengono troncati (floor) e viene generato un warning.

    Args:
        ns (npt.NDArray[np.number]): Array 1D di numeri (interi o float).

    Returns:
        Tuple[int, npt.NDArray[np.int64]]: Una tupla contenente:
            - gcd (int): Il Massimo Comune Divisore dell'array.
            - divisors (npt.NDArray[np.int64]): Array 1D dei divisori comuni
              ordinati in modo crescente. Restituisce un array vuoto se
              l'input è vuoto o se l'MCD è 0.

    Warns:
        UserWarning: Se l'array di input contiene float non esatti.
    """

    # Gestione array vuoto (blocco singolo)
    if ns.size == 0:
        return 0, np.array([], dtype=np.int64)

    # Validazione e cast sicuro
    if np.issubdtype(ns.dtype, np.floating):
        ns_int = np.floor(ns).astype(np.int64)
        if not np.array_equal(ns, ns_int):
            warnings.warn(
                "L'array contiene float non esatti. I valori sono stati troncati "
                "a intero per il calcolo dell'MCD."
            )
    else:
        ns_int = np.asarray(ns, dtype=np.int64)

    # Calcolo ultra-rapido dell'MCD su tutto l'array
    overall_gcd = np.gcd.reduce(ns_int)

    # Gestione MCD nullo (es. Array di soli zeri)
    if overall_gcd == 0:
        return 0, np.array([], dtype=np.int64)

    overall_gcd = int(abs(overall_gcd))

    # Estrazione vettorizzata dei divisori dell'MCD
    limit = math.isqrt(overall_gcd)
    candidates = np.arange(1, limit + 1, dtype=np.int64)
    divs = candidates[overall_gcd % candidates == 0]

    # Combinazione con i divisori complementari e rimozione duplicati
    return overall_gcd, np.unique(np.concatenate((divs, overall_gcd // divs)))


def find_near_divisors(number: float, error: float) -> npt.NDArray[np.float64]:
    """
    Trova i divisori di un singolo numero (float) entro un margine di tolleranza.

    Questa funzione agisce da wrapper per `common_divisors_with_tolerance`,
    incapsulando lo scalare di input in un array NumPy ed estraendo
    esclusivamente l'array dei divisori (ignorando il massimo divisore).

    Args:
        number (float): numero di cui cercare i divisori tollerati.
        error (float): Tolleranza massima assoluta ammessa per il resto.

    Returns:
        npt.NDArray[np.float64]: Array 1D dei divisori validi ordinati in modo crescente.
    """

    # Wrapper: cast dello scalare ad array 1D per riutilizzare la logica vettoriale core
    _, divs = common_divisors_with_tolerance(
        np.array([number], dtype=np.float64), error
    )
    return divs


def common_divisors_with_tolerance(
    numbers: npt.NDArray[np.float64], error: float
) -> Tuple[float, npt.NDArray[np.float64]]:
    """
    Calcola i divisori comuni di un array di float entro un margine di tolleranza.

    Sfrutta l'aritmetica modulare per valutare se il resto della divisione
    è vicino a 0 o al divisore stesso (simulando un errore fluttuante simmetrico).

    Attenzione: questo algoritmo genera una matrice intermedia di dimensione
    (min(numbers) x len(numbers)). Non è adatto a numeri di magnitudo estrema.

    Args:
        numbers (npt.NDArray[np.floating]): Array 1D di numeri positivi.
        error (float): Tolleranza massima assoluta ammessa per il resto.

    Returns:
        Tuple[float, npt.NDArray[np.float64]]:
            - max_divisor: Il divisore più grande trovato (o 1.0 come fallback).
            - divisors: Array dei divisori validi ordinati in modo crescente.
    """

    if numbers.size == 0:
        return 1.0, np.array([1.0], dtype=np.float64)

    min_val = np.min(numbers)

    # Early exit per valori sub-unitari
    if min_val < 1.0:
        return 1.0, np.array([1.0], dtype=np.float64)

    # Generazione array colonna dei candidati [1, limit]
    limit = int(np.floor(min_val))
    dividers = np.arange(1, limit + 1, dtype=np.float64).reshape(-1, 1)

    # Matrice dei resti via broadcasting (shape: limit x N)
    remains = numbers % dividers

    # Calcolo dell'errore di approssimazione circolare
    dists = np.minimum(remains, dividers - remains)

    # Maschera booleana basata sul worst-case error per divisore
    errors = np.max(dists, axis=1)
    validation_mask = errors <= error
    sure_divisors = dividers[validation_mask].flatten()

    if sure_divisors.size == 0:
        return 1.0, np.array([1.0], dtype=np.float64)

    # Estrazione del massimo (array pre-ordinato da arange)
    return float(sure_divisors[-1]), sure_divisors


def filter_by_tolerance(
    numbers: npt.NDArray[np.floating], digit: int, assume_sorted: bool = False
) -> npt.NDArray[np.float64]:
    """
    Filtra i duplicati in un array applicando una quantizzazione (arrotondamento).

    Raggruppa i numeri che differiscono meno dell'ordine di grandezza specificato.
    Se la tolleranza richiesta è troppo grande rispetto ai valori dell'array,
    viene ricalcolata automaticamente sollevando un warning.

    Args:
        numbers (npt.NDArray[np.floating]): Array 1D di numeri da filtrare.
        digit (int): Ordine di grandezza per l'arrotondamento (es. 2 per le
                     centinaia, -3 per i millesimi).
        assume_sorted (bool): Se True, assume che l'array sia già ordinato
                              in modo crescente per ottimizzare la ricerca del massimo.

    Returns:
        npt.NDArray[np.float64]: Array 1D dei valori quantizzati e univoci.
    """

    # Early exit per array vuoti
    if numbers.size == 0:
        return np.array([], dtype=np.float64)

    # Estrazione del picco massimo (in valore assoluto) per la scala
    if assume_sorted:
        max_val = max(abs(numbers[0]), abs(numbers[-1]))
    else:
        max_val = np.max(np.abs(numbers))

    # Gestione sicura dell'array di soli zeri (evita log10(0))
    if math.isclose(max_val, 0, abs_tol=10 ** (-3 - digit if digit < 10 else -digit)):
        return np.array([0.0], dtype=np.float64)

    # Fallback logico: ricalcolo dell'ordine di grandezza se eccessivo
    if max_val < 10.0**digit:
        warnings.warn(
            "Tolleranza sproporzionata. L'ordine di grandezza è stato ricalcolato in base al massimo valore dell'array."
        )
        # Shift di 2 ordini di grandezza inferiori rispetto al massimo
        digit = int(np.floor(np.log10(max_val)) - 2)

    # Quantizzazione tramite arrotondamento all'ordine di magnitudo (-digit) e filtraggio
    rounded_array = np.round(numbers, -digit)
    return np.unique(rounded_array)


def sort_by_tolerance_2d_array(
    points: npt.NDArray[np.float64], decimals: int = 7
) -> npt.NDArray[np.float64]:
    """
    Ordina un set di punti 2D in base al loro angolo polare con tolleranza.

    L'ordinamento avviene calcolando l'angolo radiale (in radianti) di ogni punto
    rispetto all'origine (0,0) utilizzando la funzione arcotangente2. Viene applicata
    una tolleranza tramite arrotondamento per gestire errori di floating-point
    e garantire un ordinamento consistente di punti quasi collineari.

    Args:
        points: Un array NumPy di forma (N, 2) contenente le coordinate [x, y].
            Supporta sottotipi di np.floating (float32, float64).
        decimals: Numero di cifre decimali a cui arrotondare l'angolo
            prima dell'ordinamento. Previene instabilità dovute a epsilon di macchina.
            Default a 8.

    Returns:
        Un nuovo array NumPy di forma (N, 2) con i punti ordinati in senso
        antiorario, partendo dall'asse X positivo.

    Raises:
        ValueError: Se l'array in input non ha la forma corretta (N, 2).
    """

    # Verifica della dimensione del array
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"Attesi punti (N, 2), ricevuto {points.shape}")
    if points.dtype != np.float64 and points.dtype != np.number:
        raise ValueError(f"Data type dell'array non valido, ricevuto {points.dtype}")

    # Calcolo angoli polari per ogni vertice
    alphas = np.arctan2(points[:, 1], points[:, 0])

    # Approssimazione a tot cifre decimale in fase di ordinamento
    if decimals is not None:
        alphas = np.round(alphas, decimals=decimals)

    # Calcolo degli indici ordinati e della copia dell'array originale
    indices = np.argsort(alphas, kind="stable")
    return points[indices]
