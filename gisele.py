import streamlit as st

# Initialize the app
st.set_page_config(layout="wide")
st.title("Local GISEle")

# Main navigation
main_nav = st.sidebar.radio("Navigation", ["Home", "Area Selection", "Data Collection", "Data Analysis"], key="main_nav")

if main_nav == "Home":
    st.write("Welcome to Local GISEle")
    st.write("Use the sidebar to navigate to different sections of the app.")
elif main_nav == "Area Selection":
    st.write("Area Selection")
    # Include your area selection logic here
elif main_nav == "Data Collection":
    # Sub-navigation for Data Collection
    data_collection_nav = st.sidebar.radio("Data Collection", ["Buildings", "Roads", "Points of Interest"], key="data_collection_nav")

    if data_collection_nav == "Buildings":
        st.write("Data Collection: Buildings")
        # Include your buildings data collection logic here
    elif data_collection_nav == "Roads":
        st.write("Data Collection: Roads")
        # Include your roads data collection logic here
    elif data_collection_nav == "Points of Interest":
        st.write("Data Collection: Points of Interest")
        # Include your points of interest data collection logic here
elif main_nav == "Data Analysis":
    st.write("Data Analysis page under construction")
    # Include your data analysis logic here

# Add About and Contact sections in the sidebar
st.sidebar.title("About")
st.sidebar.info(
    """
    Web App URL: [https://gisele.streamlit.app/](https://gisele.streamlit.app/)
    GitHub repository: [https://github.com/darlainedeme/GISEle](https://github.com/darlainedeme/GISEle)
    """
)

st.sidebar.title("Contact")
st.sidebar.info(
    """
    Darlain Edeme: [http://www.e4g.polimi.it/](http://www.e4g.polimi.it/)
    [GitHub](https://github.com/darlainedeme) | [Twitter](https://twitter.com/darlainedeme) | [LinkedIn](https://www.linkedin.com/in/darlain-edeme)
    """
)
