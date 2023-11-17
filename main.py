from fastapi import FastAPI
import pickle
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI()

# モデル（TF-IDFベクトルライザー）のロード
with open(r"C:\Users\champ\OneDrive\デスクトップ\sake\sake_tfidf_vectorizer.pkl", "rb") as file:
    vectorizer = pickle.load(file)

# 類似度行列のロード
with open(r"C:\Users\champ\OneDrive\デスクトップ\sake\sake_recommendation_model.pkl", "rb") as file:
    grouped_cosine_sim = pickle.load(file)

# データ型の定義
class SakeNameInput(BaseModel):
    sake_name: str  # sake_nameというフィールドのデータ型を定義

# 新しいGETリクエスト用エンドポイント（トップページ）
@app.get("/")
async def read_root():
    return {"message": "Welcome to the top page of the sake recommendation system!"}

# 銘柄名を受け取り、類似した銘柄名上位5つを返すAPIエンドポイント
@app.post("/recommend/")
async def get_recommendations(sake_name: str):
    # 入力銘柄名のTF-IDFベクトルを計算
    sake_vector = vectorizer.transform([sake_name])
    
    # 入力銘柄と他の銘柄との類似度スコアを計算
    sim_scores = cosine_similarity(sake_vector, grouped_cosine_sim)[0]
    
    # 類似度スコアに基づいて上位5つの銘柄を選択
    top_indices = sim_scores.argsort()[::-1][1:6]  # 類似度が高い順に並べ、最初の銘柄（自身）を除外
    
    # 上位5つの銘柄名を取得
    top_sake_names = list(grouped_cosine_sim.index[top_indices])
    
    return {"recommendations": top_sake_names}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
