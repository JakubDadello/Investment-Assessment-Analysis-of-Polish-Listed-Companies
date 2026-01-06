from pydantic import BaseModel

class Model_Request (BaseModel):
   net_income: float
   net_cash_flow: float 
   roe: float
   roa: float
   ebitda: float
   sector: str
   cumulation: int  
