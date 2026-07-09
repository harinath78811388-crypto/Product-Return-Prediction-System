import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="📦 Product Return Prediction system",page_icon="🛒",layout="wide")
st.title(" 💳Product Return Prediction System")
st.caption("Machine learning mini project 4 using decision tree classifier")
st.divider()

df=pd.read_csv("product.csv")

x=df[["Product_Price","Delivery_Days","Customer_Rating","Purchase_Count"]]
y=df["Return_Status"]

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42,stratify=y)
model=DecisionTreeClassifier(max_depth=4,random_state=42)
model.fit(x_train,y_train)
prediction = model.predict(x_test)
accuracy =accuracy_score(y_test,prediction)

col1,col2,col3 = st.columns(3)
with col1:
    st.metric("Model Accuray :",f"{accuracy*100:.2f}%")
with col2:
    st.metric("Total Records",len(df))
with col3:
    st.metric("Features",x.shape[1])
st.divider()

tab1, tab2, tab3 = st.tabs(["Dataset","Statistics","Missing value"])
with tab1:
    st.dataframe(df,use_container_width=True)
with tab2:
    st.write(df.describe())
with tab3:
    st.write(df.isnull().sum())
st.divider()

st.subheader("📃Enter product Details")
left, right = st.columns(2)
with left:
    price = st.number_input("🏷Product Price", min_value=100, max_value=3000, value=1000, step=100)
    rating =st.slider("🎇Costomer Rating",1,5,4)
with right:
    days = st.slider("✨Delivery Days",1,15,5)
    Purchese = st.number_input("🛒Purchese count",min_value=1,max_value=20,value=5,step=1)
st.divider()
if st.button("✋Predict Return Status",use_container_width=True):
    result = model.predict([[price,days,rating,Purchese]])
    st.divider()
    col1, col2  = st.columns(2)
    with col1:
        st.subheader("Prediction Result")
        if result[0]==1:
            st.error("🎗 Your Product will be returned")
            st.warning("Reason: \n"
            "- Customer Rating is low. \n"
            "- Delivery time  is high. \n"
            "- Purchese frequency is low")
        else:
            st.success("Product will not be returned")
            st.info("Reason: \n"
            "- Customer Rating is good. \n"
            "- Delivery is fast. \n"
            "- Customer frequently purchese product")
    with col2:
        st.subheader("Entered Details")
        st.write(f"**Product Price :**₹{price:,}")
        st.write(f"**Delivery Days :** {days} Days")
        st.write(f"**rating :** {rating}✨")
        st.write(f"**Purchase count :** {Purchese}")
st.divider()
st.divider()
csv = df.to_csv(index=False)
st.download_button(label="⬇️ Download Dataset",data=csv,file_name="product1.csv",mime="text/csv",use_container_width=True)

with st.expander("📖 Project Information "):
    st.write("""
    ### Project Return Prediction
    This Project whether customer is likely to return  product
    ### Algorithm Used
    - Decision Tree Classifier
    ### Input Feacture
    - Product Price
    - Delivery Days
    - Customer Rating
    - Purchase count

    ### Output
    - Product Returned
    - Product Not Returned
    """)

st.divider()
st.caption("ML Mini Project -4 ")
st.caption("Developed By Harinath")
