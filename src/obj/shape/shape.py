from abc import ABC, abstractmethod

from matplotlib import pyplot as plt
from numpy import typing as nptyping
from typing import Tuple, List, Literal, Optional
from src.utils import *

Resets = Union[List[Literal["all", "length", "area", "bounding_box", "barycenter"]], Literal["all", "length", "area", "bounding_box", "barycenter"], None]

class Shape(ABC):
    """
    Classe base astratta per la modellazione, manipolazione e visualizzazione di geometrie 2D.

    La classe `Shape` fornisce un'infrastruttura robusta per gestire contorni chiusi (poligoni).
    È progettata con un focus sulle performance, delegando alle classi figlie la sola logica
    di generazione dei punti (discretizzazione) e centralizzando i calcoli matematici complessi.

    **Funzionalità Principali:**

    * **Lazy Evaluation (Caching):** Le proprietà geometriche computazionalmente intensive
      (come `area`, `length`, `barycenter`, `bounding_box`) vengono calcolate solo al primo
      accesso e salvate in cache. Le trasformazioni invalidano automaticamente la cache
      in modo selettivo (es. una traslazione non ricalcola l'area).
    * **Trasformazioni Spaziali:** Supporto nativo per isometrie (metodi `translate` e `rotate`)
      e trasformazioni di similitudine (metodo `scale`), con aggiornamento automatico del contorno.
    * **Visualizzazione Integrata:** Metodi integrati (`draw`, `style_graph`) per il rendering
      immediato della forma e dell'origine tramite Matplotlib.

    **Regole per l'Ereditarietà (Subclassing):**

    Essendo una classe astratta, qualsiasi geometria derivata (es. `Cerchio`, `Rettangolo`)
    deve obbligatoriamente implementare i seguenti metodi per gestire il campionamento dei punti:

    1. `_discretization(self)`: Logica matematica per generare il contorno.
    2. `is_valid_step(self, custom_step: float) -> bool`: Regole di validazione del passo.
    3. `calc_min_closure_step(self, points: ArrayLike) -> float`: Calcolo del passo minimo.
    4. `calc_max_closure_step(self, points: ArrayLike) -> float`: Calcolo del passo massimo.
    5. `@closure_step.setter`: Logica di assegnazione sicura della risoluzione.
    """

    #   **Esempio di Utilizzo:**
    #       >>> mia_forma = MiaFormaDerivata(raggio=5)
    #       >>> mia_forma.discretize(custom_step=0.1)
    #       >>> mia_forma.translate(offset=(10, 5))
    #       >>> area = mia_forma.area  # Calcolata e messa in cache qui
    #       >>> mia_forma.draw(show=True)

    def __init__(self):
        """
        Inizializza l'istanza predisponendo la cache per le proprietà complesse.

        Tutti gli attributi computazionalmente intensivi (come baricentro, area e bounding box) vengono inizializzati a `None`.
        Verranno calcolati e salvati in cache (lazy evaluation) solo al momento del primo accesso tramite le rispettive proprietà pubbliche:
            - Proprietà fisiche: area, lunghezza, baricentro.
            - Metriche spaziali: bounding box.
            - Dati di chiusura: array di closure, step, step min e max..
        """
        self._origin: nptyping.NDArray[np.float64] = np.zeros(2).flatten()
        self._origin_is_center: bool = False

        self._barycenter: nptyping.NDArray[np.float64] | None = None
        self._length: float | None = None
        self._area: float | None = None
        self._bounding_box: Tuple[float, float, float, float, float, float] | None = None

        self._closure: nptyping.NDArray[np.float64] | None = None
        self._closure_step: float | None = None
        self._closure_step_min: float | None = None
        self._closure_step_max: float | None = None

    @property
    def origin(self) -> nptyping.NDArray[np.float64]:
        """
        Punto di origine dell'entità geometrica.

        :return: Array numpy 2D (x, y) che fa da **riferimento per le trasformazioni** spaziali (default = (0, 0)).
        :rtype: numpy.typing.NDArray[np.float64]
        """
        return self._origin
    @property
    def origin_is_center(self) -> bool:
        """
        Verifica se l'origine coincide con il baricentro.

        :return: **True** se il riferimento per le trasformazioni è il **centro geometrico** della forma, **False** altrimenti.
        :rtype: bool
        """
        return self._origin_is_center
    @origin.setter
    def origin(self, origin: ArrayLike):
        """
        Imposta una nuova origine per l'entità geometrica e aggiorna lo stato del centro.

        :param origin: Le nuove coordinate dell'origine. Può essere qualsiasi oggetto convertibile in un array NumPy (es. lista, tupla, ndarray).
        :type origin: ArrayLike
        :raises ValueError: Se la dimensione dell'array fornito non corrisponde a quella attesa per l'origine.
        """
        no = np.asarray(origin, dtype=np.float64, copy=True).flatten()
        if no.shape != self._origin.shape:
            raise ValueError(f"Dimensione errata: attesa {self._origin.shape}, ricevuta {no.shape}")

        self._origin = no
        self._set_origin_is_center()
    def _set_origin_is_center(self):
        """
        Ricalcola il flag interno che indica se l'origine coincide con il baricentro.

        Metodo privato (protected). Verifica se la forma ha una chiusura valida: in caso affermativo,
        confronta l'**origine** con il **baricentro** usando una tolleranza predefinita (`Eps.eps08`).
        In caso di geometria non valida o errori durante il calcolo, il flag viene impostato a **False**.
        """
        try:
            if self._is_valid_closure():
                self._origin_is_center = np.allclose(self.origin, self.barycenter, atol=Eps.eps08)
            else:
                self._origin_is_center = False
        except ValueError:
            self._origin_is_center = False

    @property
    def area(self) -> float:
        """
        Restituisce il valore assoluto dell'area della geometria.

        Sfrutta la stessa **cache** della proprietà `area`. Se l'area non è ancora stata
        calcolata, invoca la validazione e il calcolo, per poi restituirne il valore assoluto.

        :return: L'area della forma geometrica, strettamente positiva.
        :rtype: float
        :raises ValueError: Se la geometria non è valida o ha meno di 3 punti.
        """
        if self._area is None:
            self._ensure_property("area")
            self._area = Shape.calc_area(self.closure)
        return abs(self._area)
    @property
    def sign_area(self) -> float:
        """
        Restituisce l'area algebrica (con potenziale segno) della geometria.

        Il calcolo utilizza la **lazy evaluation**: viene eseguito solo al primo accesso
        tramite `Shape.calc_area` e salvato in cache per le chiamate successive.

        :return: Il valore dell'area (può essere negativo in base all'orientamento dei vertici).
        :rtype: float
        :raises ValueError: Se la geometria non è valida o ha meno di 3 punti (via `_ensure_property`).
        """
        if self._area is None:
            self._ensure_property("area con segno")
            self._area = Shape.calc_area(self.closure)
        return self._area
    @property
    def length(self) -> float:
        """
        Restituisce il valore assoluto dell'area della geometria.

        Sfrutta la stessa **cache** della proprietà `area`. Se l'area non è ancora stata
        calcolata, invoca la validazione e il calcolo, per poi restituirne il valore assoluto.

        :return: L'area della forma geometrica, strettamente positiva.
        :rtype: float
        :raises ValueError: Se la geometria non è valida o ha meno di 3 punti.
        """
        if self._length is None:
            self._ensure_property("lunghezza")
            self._length = Shape.calc_length(self.closure)
        return self._length
    @property
    def barycenter(self) -> nptyping.NDArray[np.float64]:
        """
        Restituisce le coordinate del baricentro (centro geometrico) della forma.

        Valore calcolato tramite **lazy evaluation** al primo accesso e salvato in cache.

        :return: Un array 2D contenente le coordinate `[x, y]` del baricentro.
        :rtype: nptyping.NDArray[np.float64]
        :raises ValueError: Se la geometria non è sufficientemente definita per il calcolo.
        """
        if self._barycenter is None:
            self._ensure_property("baricentro")
            self._barycenter = Shape.calc_barycenter(self.closure)
        return self._barycenter
    @property
    def bounding_box(self) -> Tuple[float, float, float, float, float, float]:
        """
        Calcola il Bounding Box (scatola di delimitazione) della geometria.

        Determina i limiti spaziali minimi e massimi occupati dalla forma.
        Il risultato è salvato in **cache** dopo il primo calcolo.

        :return: Una tupla di 6 elementi che rappresenta i limiti del bounding box (es. min_x, max_x, min_y, max_y, min_z, max_z).
        :rtype: Tuple[float, float, float, float, float, float]
        :raises ValueError: Se i punti della chiusura non sono validi.
        """
        if self._bounding_box is None:
            self._ensure_property("bounding box")
            self._bounding_box = Shape.calc_bounding_box(self.closure)
        return self._bounding_box

    @staticmethod
    def calc_length(points: ArrayLike) -> float:
        """
        Calcola il perimetro (o lunghezza totale) del poligono formato dai punti.

        L'algoritmo calcola la distanza euclidea tra ogni coppia di punti consecutivi,
        incluso il segmento che chiude la forma (dall'ultimo al primo punto, gestito
        tramite lo shift dell'array con `np.roll`).

        :param points: Le coordinate dei vertici del poligono.
        :type points: ArrayLike
        :return: La somma delle distanze tra i punti (lunghezza totale).
        :rtype: float
        :raises ValueError: Se l'array contiene meno di 3 vertici.
        """
        points_arr = np.asarray(points, dtype=np.float64)
        if len(points_arr) < 3:
            raise ValueError("Impossibile calcolare il perimetro: vertici insufficienti")

        diffs = points_arr - np.roll(points_arr, -1, axis=0)
        dists = np.linalg.norm(diffs, axis=1)
        return np.sum(dists)
    @staticmethod
    def calc_area(points: ArrayLike) -> float:
        """
        Calcola l'area algebrica del poligono definito dai punti.

        Utilizza la formula dell'area di Gauss (nota anche come formula dei lacci o *Shoelace formula*):
        $$A = \\frac{1}{2} \\sum_{i=1}^{n} (x_i y_{i+1} - x_{i+1} y_i)$$
        Il segno del risultato dipende dall'ordine dei vertici: sarà positivo per
        poligoni orientati in senso antiorario e negativo per quelli in senso orario.

        :param points: Le coordinate dei vertici del poligono.
        :type points: ArrayLike
        :return: L'area con segno della geometria.
        :rtype: float
        :raises ValueError: Se l'array contiene meno di 3 vertici.
        """
        points_arr = np.asarray(points, dtype=np.float64)
        if len(points_arr) < 3:
            raise ValueError("Impossibile calcolare l'area: vertici insufficienti")

        x = points_arr[:, 0]
        y = points_arr[:, 1]

        x_next = np.roll(x, -1)
        y_next = np.roll(y, -1)

        double_area = np.sum(x * y_next - x_next * y)
        return 0.5 * double_area
    @staticmethod
    def calc_bounding_box(points: ArrayLike) -> Tuple[float, float, float, float, float, float]:
        """
        Calcola il bounding box (limiti spaziali) per l'insieme di punti fornito.

        Identifica i valori minimi e massimi lungo gli assi X e Y. Attualmente,
        i limiti per l'orientamento (theta) sono fissati in modo statico a 0.0.

        :param points: Le coordinate dei vertici della geometria.
        :type points: ArrayLike
        :return: Una tupla di 6 elementi `(x_min, x_max, y_min, y_max, theta_min, theta_max)`.
        :rtype: Tuple[float, float, float, float, float, float]
        :raises ValueError: Se l'array contiene meno di 3 vertici.
        """
        points_arr = np.asarray(points)
        if len(points_arr) < 3:
            raise ValueError("Impossibile calcolare il boundary: vertici insufficienti")

        x = points_arr[:, 0]
        y = points_arr[:, 1]

        x_min, x_max = np.min(x), np.max(x)
        y_min, y_max = np.min(y), np.max(y)

        theta_min, theta_max = 0.0, 0.0

        new_bounding = (x_min, x_max, y_min, y_max, theta_min, theta_max)
        return new_bounding
    @staticmethod
    def calc_barycenter(points: ArrayLike) -> nptyping.NDArray[np.float64]:
        """
        Calcola il centroide (baricentro) dei vertici del poligono.

        Il calcolo viene eseguito come semplice media aritmetica delle coordinate
        dei punti lungo ciascun asse.

        :param points: Le coordinate dei vertici.
        :type points: ArrayLike
        :return: Un array 1D contenente le coordinate `[x, y]` del baricentro.
        :rtype: nptyping.NDArray[np.float64]
        :raises ValueError: Se l'array contiene meno di 3 vertici.
        """
        points_arr = np.asarray(points)
        if len(points_arr) < 3:
            raise ValueError("Impossibile calcolare il baricentro: vertici insufficienti")
        return np.mean(points_arr, axis=0)

    @property
    def closure(self) -> nptyping.NDArray[np.float64] | None:
        """
        Coordinate dei vertici che definiscono il perimetro chiuso della geometria.

        Restituisce l'array 2D dei punti attualmente in memoria. Se la forma
        non è stata ancora inizializzata con dei dati validi, restituisce **None**.

        :return: Array 2D delle coordinate `[x, y]` della chiusura, oppure **None**.
        :rtype: nptyping.NDArray[np.float64] | None
        """
        return self._closure
    @property
    def closure_step(self) -> float | None:
        """
        Passo di campionamento o discretizzazione della chiusura.

        Rappresenta la risoluzione o la distanza target attuale (se definita) con cui
        è stata generata o processata la geometria.

        :return: Il valore del passo di chiusura, oppure **None** se non è stato specificato.
        :rtype: float | None
        """
        return self._closure_step
    @property
    def closure_step_min(self) -> float:
        """
        Calcola la distanza minima tra due vertici adiacenti del poligono.

        Il calcolo utilizza la **lazy evaluation**: viene eseguito solo al primo accesso
        tramite `calc_min_closure_step` e salvato in **cache** per ottimizzare le chiamate successive.

        :return: La lunghezza del segmento più corto che compone la chiusura.
        :rtype: float
        :raises ValueError: Se i punti della geometria non sono validi o sono meno di 3 (gestito da `_ensure_property`).
        """
        if self._closure_step_min is None:
            self._ensure_property("step_minimo")
            self._closure_step_min = self.calc_min_closure_step(self.closure)
        return self._closure_step_min
    @property
    def closure_step_max(self) -> float:
        """
        Calcola la distanza massima tra due vertici adiacenti del poligono.

        Il calcolo utilizza la **lazy evaluation**: viene eseguito solo al primo accesso
        tramite `calc_max_closure_step` e salvato in **cache** per ottimizzare le chiamate successive.

        :return: La lunghezza del segmento più lungo che compone la chiusura.
        :rtype: float
        :raises ValueError: Se i punti della geometria non sono validi o sono meno di 3.
        """
        if self._closure_step_max is None:
            self._ensure_property("step_massimo")
            self._closure_step_max = self.calc_max_closure_step(self.closure)
        return self._closure_step_max

    def discretize(self, custom_step: Optional[float] = None):
        """
        Esegue la discretizzazione della geometria e aggiorna i punti della chiusura.

        Se viene fornito un `custom_step` compatibile (valutato tramite tolleranza `Eps.eps08`),
        tenta di aggiornare il passo di chiusura. Successivamente, invoca la logica
        interna di discretizzazione (`_discretization`) e **svuota la cache** per
        forzare il ricalcolo delle proprietà geometriche (area, baricentro, ecc.).

        :param custom_step: Il nuovo passo di discretizzazione desiderato. Se **None**, viene utilizzato il passo preimpostato.
        :type custom_step: Optional[float]
        :raises UserWarning: Emette un warning (non blocca l'esecuzione) se la discretizzazione produce una geometria non valida (es. meno di 3 punti).
        """
        if np.isclose(custom_step, self._closure_step, atol=Eps.eps08):
            try:
                self.closure_step = custom_step
            except (TypeError, ValueError):
                pass

        self._discretization()
        if not self._is_valid_closure():
            warnings.warn("La discretizzazione non ha prodotto un contorno valido (almeno 3 punti).")

        self.reset_cache()
    @abstractmethod
    def _discretization(self):
        """
        Implementa la logica matematica per generare i punti discretizzati.

        **Metodo Astratto**: Tutte le classi figlie devono sovrascrivere questo metodo.
        L'implementazione deve generare le nuove coordinate della forma basandosi
        sull'attuale `closure_step` e assegnarle alla variabile interna della chiusura.
        """
        pass

    @abstractmethod
    def calc_min_closure_step(self, points: ArrayLike) -> float:
        """
        Calcola la distanza minima tra i vertici consecutivi della geometria.

        **Metodo Astratto**: Da implementare nelle classi figlie.

        :param points: L'array delle coordinate dei vertici da valutare.
        :type points: ArrayLike
        :return: La distanza minima calcolata tra due punti adiacenti.
        :rtype: float
        """
        pass
    @abstractmethod
    def calc_max_closure_step(self, points: ArrayLike) -> float:
        """
        Calcola la distanza massima tra i vertici consecutivi della geometria.

        **Metodo Astratto**: Da implementare nelle classi figlie.

        :param points: L'array delle coordinate dei vertici da valutare.
        :type points: ArrayLike
        :return: La distanza massima calcolata tra due punti adiacenti.
        :rtype: float
        """
        pass

    @closure_step.setter
    @abstractmethod
    def closure_step(self, closure_step: float) -> None:
        """
        Imposta il nuovo passo di discretizzazione per la forma geometrica.

        **Metodo Astratto**: Le classi figlie devono implementare i controlli di
        validazione specifici per il passo (es. assicurarsi che sia > 0) e
        assegnare il valore alla variabile interna.

        :param closure_step: Il valore del nuovo passo di discretizzazione.
        :type closure_step: float
        """
        pass

    def _ensure_property(self, value: str):
        """
        Verifica che i dati geometrici siano sufficienti prima di calcolare una proprietà.

        Questo metodo di supporto viene chiamato prima del calcolo effettivo delle proprietà in cache (come area o baricentro) per garantire che la forma sia ben definita.

        :param value: Il nome della proprietà che si sta tentando di calcolare (es. 'area' o 'barycenter'). Utilizzato per formattare il messaggio di errore.
        :type value: str
        :raises ValueError: Se la chiusura non è valida (es. contiene meno di 3 punti o è assente).
        """
        if not self._is_valid_closure():
            n_points = len(self._closure) if self._closure is not None else 0
            raise ValueError(f"Impossibile calcolare {value}: servono almeno 3 punti (trovati: {n_points}).")
    def _is_valid_closure(self) -> bool:
        """
        Verifica la validità strutturale dell'array delle coordinate di chiusura.

        Tenta di validare l'attuale geometria usando `validate_2d_coordinates`.

        :return: **True** se la chiusura è valida. Restituisce **False** se le coordinate non sono convertibili in un array NumPy (array-like) o se non hanno l'esatta dimensione (2,).
        :rtype: bool
        """
        try :
            result = validate_array_of_2d_coordinates(self.closure)
            if result is not None:
                self._closure = result
            return True
        except (ValueError, TypeError) :
            return False

    @abstractmethod
    def is_valid_step(self, custom_step: float) -> bool:
        """
        Verifica se un determinato passo di discretizzazione è valido per la geometria.

        **Metodo Astratto**: Da implementare nelle classi figlie.
        L'implementazione deve definire le regole specifiche per l'accettazione del passo
        (es. controllo sui valori negativi o nulli, limiti massimi basati sulle dimensioni della forma).

        :param custom_step: Il valore del passo di discretizzazione da validare.
        :type custom_step: float
        :return: **True** se il passo fornito rispetta i vincoli della geometria ed è applicabile, **False** altrimenti.
        :rtype: bool
        """
        pass

    def translate(self, offset: Tuple[float, float] | ArrayLike = (0, 0)):
        """
        Applica una traslazione spaziale all'intera geometria.

        Modifica i punti della chiusura spostandoli lungo il vettore fornito.
        Al termine dell'operazione, invalida la **cache** per il bounding box e il baricentro.

        :param offset: Vettore di traslazione 2D `(dx, dy)`. Di default è (0, 0).
        :type offset: Tuple[float, float] | ArrayLike
        :raises ValueError: Se il contorno della forma non è ancora stato definito.
        """
        if not self._is_valid_closure():
            raise ValueError("Impossibile traslare: contorno non definito")

        '''
        result = validate_2d_coordinates(self.origin)
        if result is not None:
            self.origin = result
            '''

        self._closure = Shape.translate_points(self.closure, offset)
        self.reset(["bounding_box", "area", "barycenter"])
    def rotate(self, angle: float = 0, is_radiant: bool = True):
        """
        Ruota la geometria attorno al suo punto di origine.

        Se l'origine della forma non è validamente impostata, viene generato un **warning** e la rotazione avviene rispetto al centro del sistema di riferimento `(0, 0)`.
        Invalida la cache spaziale al termine dell'operazione.

        :param angle: L'angolo di rotazione.
        :type angle: float
        :param is_radiant: Se **True**, l'angolo è interpretato in radianti. Se **False**, in gradi.
        :type is_radiant: bool
        :raises ValueError: Se il contorno della forma non è ancora stato definito.
        """
        if not self._is_valid_closure():
            raise ValueError("Impossibile rotate: contorno non definito")

        try:
            result = validate_2d_coordinates(self.origin)
            if result is not None:
                self.origin = result

            ref = self.origin
        except TypeError :
            warnings.warn("Origine non definita. Verrà utilizzato (0,0) come riferimento.")
            ref = np.zeros(2)

        self._closure = Shape.rotate_points(self.closure, ref, angle, is_radiant)
        self.reset(["bounding_box", "area", "barycenter"])
    def scale(self, factor: Tuple[float, float] | ArrayLike | float = 1):
        """
        Applica una scalatura (uniforme o non uniforme) alla geometria.

        La scalatura avviene rispetto all'origine della forma (se indefinita, usa `(0, 0)`).
        Poiché questa operazione modifica le dimensioni fisiche dell'oggetto,
        viene invocato un **reset totale della cache** (area, perimetro, ecc.).

        :param factor: Fattore di scala. Può essere uno scalare (scalatura uniforme) o un array/tupla di due elementi `(sx, sy)` (scalatura non uniforme).
        :type factor: Tuple[float, float] | ArrayLike | float
        :raises ValueError: Se il contorno della forma non è ancora stato definito.
        """
        if not self._is_valid_closure():
            raise ValueError("Impossibile scalare: contorno non definito")

        try:
            result = validate_2d_coordinates(self.origin)
            if result is not None:
                self.origin = result

            ref = self.origin
        except TypeError:
            warnings.warn("Origine non definita. Verrà utilizzato (0,0) come riferimento.")
            ref = np.zeros(2)

        self._closure = Shape.scale_points(self.closure, ref, factor)
        self.reset_cache()

    @staticmethod
    def translate_points(points: ArrayLike, offset: Tuple[float, float] | ArrayLike = (0, 0)) -> nptyping.NDArray[np.float64]:
        """
        Applica una traslazione a un set di coordinate 2D.

        Esegue l'operazione vettoriale $P_{new} = P + \vec{v}$, dove $\vec{v}$ è l'offset.

        :param points: L'array dei punti da traslare.
        :type points: ArrayLike
        :param offset: Il vettore 2D `(dx, dy)` di traslazione.
        :type offset: Tuple[float, float] | ArrayLike
        :return: Un nuovo array contenente le coordinate traslate.
        :rtype: nptyping.NDArray[np.float64]
        :raises ValueError: Se l'offset non ha dimensione esattamente pari a (2,).
        """
        delta = np.asarray(offset, dtype=np.float64).flatten()
        coords = validate_2d_coordinates(points)

        if delta.shape != (2,):
            raise ValueError("L'offset deve essere un vettore di 2 elementi (dx, dy)")

        return coords + delta
    @staticmethod
    def rotate_points(points: ArrayLike, ref: ArrayLike, angle: float = 0, is_radiant: bool = True) -> nptyping.NDArray[np.float64]:
        """
        Ruota un set di punti 2D attorno a un punto di riferimento specifico.

        Applica la seguente matrice di trasformazione rispetto al punto $R(x_{ref}, y_{ref})$:
        $$x_{new} = (x - x_{ref}) \cos(\theta) - (y - y_{ref}) \sin(\theta) + x_{ref}$$
        $$y_{new} = (x - x_{ref}) \sin(\theta) + (y - y_{ref}) \cos(\theta) + y_{ref}$$

        Se l'angolo (in valore assoluto) è prossimo a zero (valutato tramite `Eps.eps08`),
        la trasformazione viene saltata per ottimizzare le performance.

        :param points: L'array dei punti da ruotare.
        :type points: ArrayLike
        :param ref: Le coordinate del punto perno della rotazione.
        :type ref: ArrayLike
        :param angle: Angolo di rotazione.
        :type angle: float
        :param is_radiant: Indica se l'angolo è in radianti (**True**) o gradi (**False**).
        :type is_radiant: bool
        :return: Un nuovo array contenente le coordinate ruotate.
        :rtype: nptyping.NDArray[np.float64]
        """
        angle_rad = angle if is_radiant else np.deg2rad(angle)
        coords = validate_2d_coordinates(points)

        if np.isclose(angle_rad, 0.0, atol=Eps.eps08):
            return coords.copy()

        c, s = np.cos(angle_rad), np.sin(angle_rad)

        centered = coords - ref
        x_new = centered[:, 0] * c - centered[:, 1] * s
        y_new = centered[:, 0] * s + centered[:, 1] * c
        centered[:, 0] = x_new
        centered[:, 1] = y_new
        centered += ref
        return centered
    @staticmethod
    def scale_points(points: ArrayLike, ref: ArrayLike, factor: Tuple[float, float] | ArrayLike | float = 1) -> nptyping.NDArray[np.float64]:
        """
        Scala un set di coordinate 2D rispetto a un punto di riferimento.

        Applica l'equazione vettoriale $P_{new} = (P - R) \cdot S + R$. Supporta sia
        scalature uniformi (dove $S$ è uno scalare) sia scalature non uniformi
        lungo gli assi X e Y (dove $S$ è un array `(sx, sy)`).

        :param points: L'array dei punti da scalare.
        :type points: ArrayLike
        :param ref: Il punto di riferimento che rimarrà invariato durante la scalatura.
        :type ref: ArrayLike
        :param factor: Il moltiplicatore di scala. Può essere un numero singolo o una coppia.
        :type factor: Tuple[float, float] | ArrayLike | float
        :return: Un nuovo array contenente le coordinate scalate.
        :rtype: nptyping.NDArray[np.float64]
        :raises ValueError: Se il fattore di scala è un array con formato non compatibile.
        :raises UserWarning: Emette un avviso se uno o entrambi i fattori di scala sono prossimi allo zero (la geometria collassa in una linea o in un punto).
        """
        s = np.asarray(factor, dtype=np.float64)
        coords = validate_2d_coordinates(points)

        if s.ndim == 1 and s.size == 2:
            s = s.reshape(1, 2)
        elif s.ndim == 0:
            pass
        elif s.shape != (1, 2):
            raise ValueError(f"Fattore di scala non valido. Atteso scalare o array (2,), ricevuto {s.shape}")

        if np.allclose(s, 1.0, atol=1e-12):
            return coords.copy()

        if np.any(np.isclose(s, 0.0, atol=1e-12)):
            warnings.warn("Scaling con fattore zero: la geometria collasserà.")

        return (coords - ref) * s + ref

    def reset_cache(self):
        """
        Svuota in modo incondizionato la cache delle proprietà geometriche principali.

        Imposta a **None** le variabili interne relative ad area, lunghezza (perimetro),
        bounding box e baricentro. Questo forzerà la classe a ricalcolare i valori
        (tramite *lazy evaluation*) al successivo accesso alle rispettive proprietà.
        """
        self._area = None
        self._length = None
        self._bounding_box = None
        self._barycenter = None
    def reset(self, resets: Resets):
        """
        Permette l'invalidazione selettiva o totale della cache.

        Utile quando una trasformazione modifica solo alcune caratteristiche della
        geometria (es. una traslazione cambia il baricentro ma non l'area o la lunghezza),
        evitando ricalcoli inutili.

        :param resets: Un iterabile contenente le chiavi delle proprietà da resettare.
                       Se vale **None** o contiene la stringa **"all"**, viene invocato `reset_cache()`
                       per svuotare tutto. Altrimenti, accetta valori specifici come
                       **"length"**, **"area"**, **"boundary"** o **"barycenter"**.
        :type resets: Resets
        """
        if resets is None:
            self.reset_cache()
            return

        for target in resets:
            if target == "all":
                self.reset_cache()
                return

            elif target == "length":
                self._length = None
            elif target == "area":
                self._area = None
            elif target == "boundary":
                self._bounding_box = None
            elif target == "barycenter":
                self._barycenter = None

    @staticmethod
    def style_graph() -> plt.Axes:
        """
        Inizializza e formatta un nuovo piano cartesiano per il disegno geometrico.

        Configura i parametri globali di stile di Matplotlib (font, griglia, spessore linee)
        e crea una nuova figura con assi aventi **proporzioni uguali** (`aspect='equal'`),
        fondamentale per non distorcere visivamente le forme geometriche.

        :return: L'oggetto assi (`plt.Axes`) pronto per essere disegnato.
        :rtype: plt.Axes
        """
        style_params = {
            'font.family': 'serif',
            'axes.grid': True,
            'grid.alpha': 0.3,
            'lines.linewidth': 1.5
        }
        plt.rcParams.update(style_params)

        fig, ax = plt.subplots(figsize=(6, 6), layout='constrained')
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.set_aspect('equal')
        return ax
    def draw(self, ax: plt.Axes = None, points: ArrayLike = None, show: bool = False, **kwargs) -> plt.Axes | None:
        """
        Renderizza la forma geometrica e l'origine su un piano cartesiano.

        Disegna il perimetro chiuso della geometria. Se non viene fornito un oggetto assi (`ax`),
        ne genera uno nuovo tramite `style_graph()`. Permette anche di plottare un
        set opzionale di punti ausiliari.

        :param ax: L'asse Matplotlib su cui disegnare. Se **None**, ne viene creato uno nuovo.
        :type ax: plt.Axes, opzionale
        :param points: Un array di coordinate 2D ausiliarie da sovrapporre al grafico (disegnate in verde).
        :type points: ArrayLike, opzionale
        :param show: Se **True**, invoca `plt.show()` al termine del disegno per mostrare immediatamente la finestra.
        :type show: bool
        :param kwargs: Argomenti addizionali passati direttamente a `ax.plot` per personalizzare la linea del contorno (es. `color='blue'`, `linewidth=2`).
        :return: L'oggetto assi utilizzato per il disegno, utile per sovrapporre ulteriori grafici, oppure **None** in caso di errore critico.
        :rtype: plt.Axes | None
        """
        ax = Shape.style_graph() if ax is None else ax
        if self._is_valid_closure():
            warnings.warn("Nessun contorno disponibile per il disegno.")  # Al posto di print()
            return ax

        try:
            if points is None:
                other_coords = None
            else:
                other_coords = validate_2d_coordinates(points)
        except (ValueError, TypeError):
            other_coords = None
        if other_coords is not None:
            x_other = other_coords[:, 0]
            y_other = other_coords[:, 1]

            ax.scatter(x_other, y_other, color="green", marker="o", label="Other Points")

        points_closed = np.vstack([self.closure, self.closure[0]])
        x, y = points_closed[:, 0], points_closed[:, 1]
        if not kwargs:
            kwargs = {'color': 'black', 'marker': 'o', 'markersize': 2}

        ax.plot(x, y, **kwargs)
        ax.plot(self.origin[0], self.origin[1], 'rx', label='Origin')

        if show:
            plt.show()

        return ax