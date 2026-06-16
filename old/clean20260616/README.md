
from rich.pretty import pprint
from rich.console import Console
from rich.pretty import Pretty
from rich.text import Text
from rich.cells import cell_len
from rich.style import Style
import pyfiglet

console = Console()
console.print(Text(pyfiglet.figlet_format("using-my-version-of-gpipe", font="slant"), style="bold cyan"))
