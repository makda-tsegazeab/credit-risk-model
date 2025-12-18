from pydantic import BaseModel

class PredictionRequest(BaseModel):
    amount_count: float
    amount_sum: float
    amount_mean: float
    amount_std: float
    amount_min: float
    amount_max: float
    amount_median: float
    most_common_CurrencyCode: int
    most_common_ProviderId: int
    most_common_ProductId: int
    most_common_ProductCategory: int
    most_common_ChannelId: int

class PredictionResponse(BaseModel):
    risk_probability: float
    risk_label: int
