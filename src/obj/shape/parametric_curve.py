from abc import ABC, abstractmethod
from src.obj.shape.shape import Shape

from matplotlib import pyplot as plt
from numpy import typing as nptyping
from src.utils import ArrayLike

class ParametricCurve(Shape, ABC):
    """
    Classe base astratta per la modellazione di geometrie basate su curve parametriche 2D.

    Estende la classe `Shape` per rappresentare forme (chiuse o aperte) governate da equazioni matematiche continue. Una curva parametrica è definita da un parametro `t` appartenente a un dominio specifico (es. da $0$ a $2\pi$ per una circonferenza).

    Le classi figlie (es. Cerchio, Ellisse, Spline) devono fornire le equazioni parametriche implementando i metodi `point_at` e definendo il dominio `t_range`.
    Il campionamento (discretizzazione) per generare il contorno effettivo sfrutterà queste equazioni.
    """
    def __init__(self):
        """
        Inizializza la curva parametrica richiamando il costruttore della classe base.
        Predispone l'infrastruttura di **lazy evaluation** e le variabili di stato.
        """
        super().__init__()

    @abstractmethod
    def point_at(self, t: ArrayLike) -> nptyping.NDArray[np.float64]:
        """
        Valuta le equazioni della curva per ottenere le coordinate spaziali.

        **Metodo Astratto**: Tutte le curve parametriche devono implementare la propria logica matematica qui. Per massimizzare le performance con NumPy, si consiglia di implementare questo metodo in modo vettorializzato (accettando un array di $t$).

        :param t: Il parametro (o l'array di parametri) in corrispondenza del quale valutare la curva. I valori dovrebbero rientrare in `t_range`.
        :type t: ArrayLike
        :return: Un array 2D contenente le coordinate `[x, y]` valutate.
        :rtype: nptyping.NDArray[np.float64]
        """
        pass
    @property
    @abstractmethod
    def t_range(self) -> tuple[float, float]:
        """
        Dominio di validità del parametro della curva.

        **Proprietà Astratta**: Da implementare nelle classi figlie.

        :return: Una tupla `(t_min, t_max)` che rappresenta i limiti inferiore e superiore entro cui il parametro $t$ varia per descrivere l'intera geometria.
        :rtype: tuple[float, float]
        """
        pass

    def draw(self, ax: plt.Axes = None, points: ArrayLike = None, show: bool = False, **kwargs) -> plt.Axes | None:
        """
        Renderizza la curva parametrica su un piano cartesiano.

        Sovrascrive e arricchisce il metodo `draw` della classe madre `Shape`.
        Se la curva non è stata ancora campionata (ovvero se `closure` è **None**), questo metodo invoca automaticamente `discretize()` per generare i punti del contorno prima di procedere al rendering.

        :param ax: L'asse Matplotlib su cui disegnare. Se **None**, ne viene creato uno nuovo.
        :type ax: plt.Axes, opzionale
        :param points: Un array di coordinate 2D ausiliarie da plottare insieme alla curva.
        :type points: ArrayLike, opzionale
        :param show: Se **True**, invoca `plt.show()` per mostrare la finestra al termine del disegno.
        :type show: bool
        :param kwargs: Argomenti addizionali passati direttamente a `ax.plot` per lo stile della linea.
        :return: L'oggetto assi utilizzato per il disegno.
        :rtype: plt.Axes | None
        """
        if self.closure is None:
            self.discretize()

        return super().draw(ax, points, show, **kwargs)