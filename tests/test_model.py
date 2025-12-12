import pytest
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

# Import the actual functions from your files
from core.dataset import prepare_data
from core.model import create_model

def test_model_creation():
    """Test that create_model returns a Decision Tree."""
    model = create_model()
    # Check if the returned object is actually a DecisionTreeClassifier
    assert isinstance(model, DecisionTreeClassifier)

def test_data_preparation():
    """Test that prepare_data correctly splits features and target."""
    # 1. Create a dummy dataframe (mocking your CSV data)
    mock_data = {
        'temperature': [37.5, 38.0, 36.5],
        'toux': [0, 1, 0],
        'fatigue': [0, 1, 1],
        'infecte': [0, 1, 0] # Target column
    }
    df = pd.DataFrame(mock_data)

    # 2. Run your function
    X, y = prepare_data(df)

    # 3. Verify results
    # X should have 3 rows and 3 columns (temp, toux, fatigue)
    assert X.shape == (3, 3) 
    # y should have 3 rows
    assert len(y) == 3
    # Check if columns are correct
    assert list(X.columns) == ['temperature', 'toux', 'fatigue']