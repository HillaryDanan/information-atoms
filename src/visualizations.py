# Additional visualization helpers
from .atoms import HexagonalGrid
from .experiments import InformationAtomTestRunner

def create_hero_visualization():
    """Create the main hero image for README"""
    runner = InformationAtomTestRunner()
    results = runner.run_all_experiments()
    return results['shape_fig']
