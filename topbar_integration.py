import streamlit as st
from string import Template

def display_topbar(username):
    if username is not None:
        html = """
                <html>
                <div class='top-bar'>
                <div class="topbarlogo">
                <img src="https://www2.deloitte.com/content/dam/assets/logos/deloitte.svg"/>
                </div>
                <div class="topbarusername">
                <p>$code</p>
                </div>
                <div class="clear"></div>
                </div>
                <style>
                .top-bar {
                    position: relative;
                    height: 60px;
                    width: 250%;
                    background-color: #000000;
                    top: 0;
                    left: 0;
                    border-right: 1px;
                    border-left: 1px;
                    margin-top: -14%;
                    margin-left: -60%;
                    }
                .topbarlogo {
                    position: relative;
                    float: center;
                    width: 150px;
                    display: inline-block;
                    padding: 20px;
                    margin-left: 2.2%;
                    }
                .topbarusername {
                    position: absolute;
                    color: white;
                    float: center;
                    width: 10px;
                    display: inline;
                    padding: 20px;
                    margin-left: 65%;
                    }
                .clear {
                  clear: both;
                }
                </style>
                </html>
                """
        s = Template(html).safe_substitute(code=username)
        print(s)
        st.markdown(s, unsafe_allow_html=True)
    else:
        st.markdown("""<html>
                        <div class='top-bar'>
                        <img src="https://www2.deloitte.com/content/dam/assets/logos/deloitte.svg" class="topbarlogo"/>
                        <div class="clear"></div>
                        </div>
                        <style>
                        .top-bar {
                            position: relative;
                            height: 60px;
                            width: 250%;
                            background-color: #000000;
                            top: 0;
                            left: 0;
                            border-right: 1px;
                            border-left: 1px;
                            margin-top: -14%;
                            margin-left: -60%;
                            }
                        .topbarlogo {
                            position: relative;
                            float: center;
                            width: 150px;
                            display: inline-block;
                            padding: 20px;
                            margin-left: 2.2%;
                            }
                        .clear {
                          clear: both;
                        }
                        </style>
                        </html>
                        """, unsafe_allow_html=True)