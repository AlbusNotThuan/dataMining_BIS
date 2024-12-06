# Time Series Product Order Prediction

This project implements a time series prediction system for product ordering, focusing on training individual models for each product and making predictions about future order quantities.

- Individual model per product allows for product-specific patterns
- Minimum data requirement (4 weeks) ensures reliable predictions
- Trend analysis using recent data helps capture current market dynamics
- Confidence scoring helps in assessing prediction reliability

## PurchaseOrderPredictor Class

```python
class PurchaseOrderPredictor:
    """
    Singleton class for predicting product orders based on historical weekly patterns.
    Only trains models once and reuses for multiple predictions.
    """
    _instance = None
    _is_initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(PurchaseOrderPredictor, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if not PurchaseOrderPredictor._is_initialized:
            self.models = {}
            self.product_metrics = None
            self.last_training_date = None
            PurchaseOrderPredictor._is_initialized = True
```

## Training Method

```python
    def train(self, purchase_orders: pd.DataFrame) -> None:
        """Train models once for all products"""
        df = purchase_orders.copy()
        df['DueDate'] = pd.to_datetime(df['DueDate'])
        df['Week'] = df['DueDate'].dt.isocalendar().week

        # Calculate weekly metrics
        self.product_metrics = df.groupby(['ProductID', 'Week']).agg({
            'OrderQty': ['mean', 'std', 'count'],
            'UnitPrice': 'mean'
        }).reset_index()

        # Train model for each product
        for product_id in df['ProductID'].unique():
            product_data = self.product_metrics[self.product_metrics['ProductID'] == product_id]

            if len(product_data) < 4:  # Minimum weeks of data required
                continue

            X = product_data[['Week']]
            y = product_data['OrderQty']['mean']

            model = RandomForestRegressor(n_estimators=100)
            model.fit(X, y)
            self.models[product_id] = model

        self.last_training_date = datetime.now()
```

The system trains a separate Random Forest model for each unique product ID in the dataset. It first filters the data for a specific product and ensures there's sufficient historical data (at least 4 weeks) before training. The model uses the week number as the feature (X) and the mean order quantity as the target variable (y). These trained models are stored in a dictionary with product IDs as keys.

## Prediction Method

```python
    def predict_top_products(self, target_date: datetime, top_n: int = 5) -> pd.DataFrame:
        """Predict top products for given week"""
        if not self.models:
            raise ValueError("Model not trained. Call train() first.")

        target_week = target_date.isocalendar().week
        predictions = []

        for product_id, model in self.models.items():
            product_data = self.product_metrics[self.product_metrics['ProductID'] == product_id]
            pred_qty = model.predict([[target_week]])[0]
            confidence = model.score(product_data[['Week']], product_data['OrderQty']['mean'])

            recent_data = product_data.tail(12)
            trend = np.polyfit(range(len(recent_data)), recent_data['OrderQty']['mean'], 1)[0]

            predictions.append({
                'ProductID': product_id,
                'PredictedQty': round(pred_qty),
                'Confidence': confidence,
                'WeeklyTrend': trend
            })

        predictions_df = pd.DataFrame(predictions)
        predictions_df['Score'] = predictions_df['Confidence'] * (1 + predictions_df['WeeklyTrend'])
        return predictions_df.nlargest(top_n, 'Score')
```

1. Predicts the order quantity for the target week
2. Calculates a confidence score using the model's R² score
3. Analyzes recent trends by fitting a linear polynomial to the last 12 weeks of data
4. Rounds predictions to whole numbers for practical use

## Visualization Method

```python
    def visualize_predictions(self, predictions: pd.DataFrame) -> None:
        """Visualize top predictions"""
        plt.figure(figsize=(12, 6))
        sns.barplot(data=predictions, x='ProductID', y='PredictedQty')
        plt.title('Top Product Predictions by Week')
        plt.xticks(rotation=45)
        plt.show()
```

## Usage Example

```python
# Create predictor instance
predictor = PurchaseOrderPredictor()

# Load data
purchase_orders = pd.read_csv('data/purchaseOrderDetail.csv')

# Train models (only needs to be done once)
predictor.train(purchase_orders)
```
