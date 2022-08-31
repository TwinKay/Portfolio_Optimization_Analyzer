import sys
from PyQt5.QtWidgets import QPushButton, QTextEdit, QLabel, QLineEdit, QHBoxLayout, \
                            QVBoxLayout, QGroupBox, QComboBox, QGridLayout
from PyQt5.QAxContainer import QAxWidget
from pykiwoom.kiwoom import time, QWidget, Kiwoom, QApplication
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import pandas as pd
import numpy as np

kiwoom = Kiwoom()

class StockWindow_Widget(QWidget):
    def __init__(self):
        super().__init__()
        self.kiwoom = QAxWidget("KHOPENAPI.KHOpenAPICtrl.1")  # API 모듈 불러오기
        self.kiwoom.OnEventConnect.connect(self._event_connect) # 로그인 관련 이벤트


        # LAYOUT SETTING_Top
        self.Top_LayOut = QHBoxLayout()

        self.groupBox_left = QGroupBox("입력창")
        self.groupBox_middle = QGroupBox("상태진행창")
        self.groupBox_right = QGroupBox("정보창")


        # WIDGETS SETTING - Top_Left
        self.label_계좌 = QLabel("계좌", self)
        self.combo_계좌 = QComboBox()

        self.label_투자금 = QLabel("투자금", self)
        self.text_투자금 = QLineEdit("", self)
        self.text_투자금.setPlaceholderText("ex. 10000000")

        self.label_종목 = QLabel("종목", self)
        self.text_종목코드 = QLineEdit("", self)
        self.text_종목코드.setPlaceholderText("ex. 005930")

        self.btn_추가 = QPushButton("추가", self)

        self.btn_리셋 = QPushButton("리셋", self)

        self.btn_분석 = QPushButton("분석", self)

        self.btn_자동매수 = QPushButton("자동 매수", self)

        self.btn_로그인 = QPushButton("Log in", self)

        self.btn_계좌비번 = QPushButton("set pw", self)


        # WIDGETS SETTING - Top_iddle
        self.vbox_middle = QVBoxLayout()
        self.상태진행창 = QTextEdit()


        # WIDGETS SETTING - Top_Right
        self.vbox_right = QVBoxLayout()
        self.정보창 = QTextEdit()


        # Make UI
        self._setup_ui()
        # Make Slot
        self._setup_slots()


    def _setup_ui(self):
        self.setWindowTitle("포트폴리오 최적화 분석기")
        self.setGeometry(100, 100, 1500, 800)

        # self.setLayout(self.Top_LayOut)
        self.Top_LayOut.addWidget(self.groupBox_left)
        self.Top_LayOut.addWidget(self.groupBox_middle)
        self.Top_LayOut.addWidget(self.groupBox_right)

        # GROUP BOX : Top_left
        layoutGrid_left = QGridLayout()
        self.groupBox_left.setLayout(layoutGrid_left)
        layoutGrid_left.addWidget(self.label_계좌,0,0)
        layoutGrid_left.addWidget(self.combo_계좌,0,1)
        layoutGrid_left.addWidget(self.label_투자금,1,0)
        layoutGrid_left.addWidget(self.text_투자금,1,1)
        layoutGrid_left.addWidget(self.label_종목,2,0)
        layoutGrid_left.addWidget(self.text_종목코드,2,1)
        layoutGrid_left.addWidget(self.btn_추가,3,1)
        layoutGrid_left.addWidget(self.btn_리셋,4,1)
        layoutGrid_left.addWidget(self.btn_분석,5,1)
        layoutGrid_left.addWidget(self.btn_자동매수,6,1)
        layoutGrid_left.addWidget(self.btn_로그인,7,0)
        layoutGrid_left.addWidget(self.btn_계좌비번,7,1)

        # GROUP BOX : Top_MIDDLE
        self.groupBox_middle.setLayout(self.vbox_middle)
        self.vbox_middle.addWidget(self.상태진행창)

        # GROUP BOX : Top_Right
        self.groupBox_right.setLayout(self.vbox_right)
        self.vbox_right.addWidget(self.정보창)

        # Figure : Bottom_Left
        self.fig1 = plt.Figure()
        self.canvas1 = FigureCanvas(self.fig1)

        # Figure : Bottom_Right
        self.fig2 = plt.Figure()
        self.canvas2 = FigureCanvas(self.fig2)

        # LAYOUT SETTING_Bottom
        Bottom_LayOut = QHBoxLayout()
        Bottom_LayOut.addWidget(self.canvas1)
        Bottom_LayOut.addWidget(self.canvas2)

        # LAYOUT SETTING_Top_and_Bottom
        layout = QVBoxLayout()
        layout.addLayout(self.Top_LayOut)
        layout.addLayout(Bottom_LayOut)
        self.setLayout(layout)


    def _setup_slots(self):
        self.btn_추가.clicked.connect(self.btn_add_CLICKED)
        self.btn_리셋.clicked.connect(self.btn_reset_CLICKED)
        self.btn_분석.clicked.connect(self.btn_analysis_CLICKED)
        self.btn_자동매수.clicked.connect(self.btn_buy_CLICKED)
        self.btn_로그인.clicked.connect(self.btn_login_CLICKED)
        self.btn_계좌비번.clicked.connect(self.btn_login2_CLICKED)

    stock_list = []
    def btn_add_CLICKED(self):
        self.stock_list.append(self.text_종목코드.text())
        self.정보창.clear()
        self.정보창.append(self.text_종목코드.text()+"(이)가 추가되었습니다.")
        self.정보창.append("현재까지 추가된 종목 : " + ", ".join(self.stock_list))
        self.text_종목코드.clear()


    def btn_reset_CLICKED(self):
        self.stock_list.clear()
        self.정보창.clear()
        self.text_종목코드.clear()
        self.정보창.append("모든 종목이 삭제되었습니다.")


    def btn_analysis_CLICKED(self):
        self.상태진행창.clear()


        # 데이터 넣을 리스트 생성
        for i in range(len(self.stock_list)):
            globals()['df_' + str(i + 1)] = list() # 전역변수 선언 _ 리스트를 밖에서도 사용하기 위함

        #리스트에 데이터 넣기
        for i in range(len(self.stock_list)):
            df_firstblock = kiwoom.block_request("opt10081",
                                                 종목코드=self.stock_list[i],
                                                 기준일자=datetime.today(),
                                                 수정주가구분=1,
                                                 output="주식일봉차트조회",
                                                 next=0)

            globals()['df_' + str(i + 1)].append(df_firstblock)

            self.상태진행창.append(self.stock_list[i] + ' 데이터 수집 시작.. ({}~)'.format(df_firstblock.loc[0, '일자']))
            self.상태진행창.append(self.stock_list[i] + ' 데이터 수집 중.. (~{})'.format(df_firstblock.loc[len(df_firstblock) - 1, '일자']))

            # 남은 데이터 계속 넣기
            while kiwoom.tr_remained:
                df_remainblock = kiwoom.block_request("opt10081",
                                                      종목코드=self.stock_list[i],
                                                      기준일자=datetime.today(),
                                                      수정주가구분=1,
                                                      output="주식일봉차트조회",
                                                      next=2)
                globals()['df_' + str(i + 1)].append(df_remainblock)
                time.sleep(0.5)     # 0.5초마다 불러오기 위함

                self.상태진행창.append(self.stock_list[i] + ' 데이터 수집 중.. (~{})'.format(df_remainblock.loc[len(df_remainblock) - 1, '일자']))

                #특정 종목 데이터 수집 완료 시
                if kiwoom.tr_remained == False:
                    self.상태진행창.append(self.stock_list[i] + ' 데이터 수집 완료')

        #상태진행창.clear()로 인해 해당 위치에 위치
        self.상태진행창.append("포트폴리오 최적화 중...")

        # 데이터 가공 : 리스트 요소의 데이터프레임화
        for i in range(len(self.stock_list)):
            globals()['df_' + str(i + 1)] = pd.concat(globals()['df_' + str(i + 1)])
            globals()['df_' + str(i + 1)].reset_index(drop=True, inplace=True)

        # 데이터 가공 : 종목코드, 일자, 현재가(종가)만 따로 추출 후, 일자는 인덱스, 종목코드는 colums명
        for i in range(len(self.stock_list)):
            globals()['price_' + str(i + 1)] = globals()['df_' + str(i + 1)][['종목코드', '일자', '현재가']].dropna(
                    how='all')
            globals()['price_' + str(i + 1)].set_index('일자', inplace=True)
            globals()['price_' + str(i + 1)].rename(columns={'현재가': globals()['price_' + str(i + 1)].iloc[0, 0]},
                                                        inplace=True)
            globals()['price_' + str(i + 1)] = globals()['price_' + str(i + 1)].drop(['종목코드'], axis=1)

        # price_all에 모두 합치기
        price_all = []
        for i in range(len(self.stock_list)):
            price_all.append(globals()['price_' + str(i + 1)])

        price_all = pd.concat(price_all, axis=1)
        price_all = price_all.sort_index()  # 추가 가공 편리성을 위한 오름차순 정렬(인덱스 기준)

        # NaN있는 행 지우고, type변경
        df = price_all.dropna()
        df = df.astype('int')


        #데이터 분석을 위한 도구 만들기
        daily_ret = df.pct_change()  # 모든 종목 데이터의 주가상승률(1일 기준)
        annual_ret = daily_ret.mean() * 252  # 주가상승률(1년 기준)(평균 영업일 : 252일)
        daily_cov = daily_ret.cov()  # 주가상승률(1일 기준)의 공분산 행렬
        annual_cov = daily_cov * 252  # 주가상승률(1일 기준)의 공분산 행렬 X 252일

        # 포트폴리오의 주가상승률(1일 기준), 리스크, 비중, Sharpe Ratio의 공백 리스트 생성
        port_ret = []
        port_risk = []
        port_weights = []
        sharpe_ratio = []


        #무작위로 포트폴리오 추출
        for _ in range(50000):
            weights = np.random.random(len(self.stock_list))
            weights /= np.sum(weights)  #가중치 만들기

            returns = np.dot(weights, annual_ret)   # 주가상승률(1년 기준) 행렬과 가중치를 내적하여 기대 수익률 만들기

            risk = np.sqrt(np.dot(weights.T, np.dot(annual_cov, weights)))  #리스크(변동성)의 기댓값
            # 'weights의 전치행렬'과 'annual_cov와 weights를 내적한 행렬'의 내적한 행렬에 제곱근을 취하여 리스크의 기댓값 만들기

            # 포트폴리오의 주가상승률(1일 기준), 리스크, 비중, Sharpe Ratio의 공백 리스트에 요소 추가 반복
            port_ret.append(returns)
            port_risk.append(risk)
            port_weights.append(weights)
            sharpe_ratio.append(returns / risk)      # 무위험 수익률은 무시


        # 주가상승률(1일 기준), 리스크, 비중, Sharpe Ratio의 리스트를 통해 데이터 프레임 만들기
        portfolio = {'Returns': port_ret, 'Risk': port_risk, 'Sharpe': sharpe_ratio}

        for i, s in enumerate(self.stock_list):               # enumerate 함수는 인덱스와 원소로 이루어진 튜플(tuple) 만들어 줌
            portfolio[s] = [ j[i] for j in port_weights ]     # ex) '005930':[0.08120~,0.07231~,...]으로 딕셔너리에 추가

        df = pd.DataFrame(portfolio)                     # 딕셔너리로 데이터 프레임 만들기

        max_sharpe = df.loc[df['Sharpe'] == df['Sharpe'].max()]     # 최대 샤프 비율의 행 찾기

        # 최대 샤프 비율 시, 종목별 구성비 구하여 딕셔너리로 만들기
        stock_weight_dict = dict(zip(list(max_sharpe[self.stock_list].columns), #Returns,Risk,Sharpe은 분리 되도록 하기 위함
                                     max_sharpe[self.stock_list].values.flatten().tolist()))
            #array내 2차원 리스트->1차원 변환 후, array를 리스트 변환 후, 순서에 맞게 묶은 뒤, 딕셔너리 만들기

        # 정보창에 최대 샤프 비율일 때의 포트폴리오 구성비 표시하기
        self.정보창.clear()
        self.정보창.append("최대 샤프 비율 시, 포트폴리오 구성비는 다음과 같습니다.")
        for i in self.stock_list:
            self.정보창.append(str(i)+" : "+str(round(stock_weight_dict[i]*100,2))+"%")
        self.정보창.append("(소수점 셋째 자리에서 반올림됨)")


        # 그래프1 그리기
        ax = self.fig1.add_subplot(111)

        ax.scatter(df['Risk'], df['Returns'], s=2, c=df['Sharpe'], edgecolors='black',linewidth=0.01,cmap='viridis')
        ax.scatter(x=max_sharpe['Risk'], y=max_sharpe['Returns'], c='r',
                                 marker='*', s=300)
        # ax.colorbar()                                    #칼라바는 오류로 구현을 못함
        ax.grid()
        ax.set_xlabel("Risk")
        ax.set_ylabel("Expected Returns")
        ax.set_title("Portfoilo Optimization")

        self.canvas1.draw()


        # 그래프2 그리기
        weight_list = []
        for i in self.stock_list:
            weight_list.append(stock_weight_dict[i]*100)    #딕셔너리에서 values 뽑아낸 후, 100을 곱하여 weight_list에 넣기

        bx = self.fig2.add_subplot(111)

        bx.bar(self.stock_list, weight_list)

        bx.grid()
        bx.set_xlabel("Stock Code(Ticker)")
        bx.set_ylabel("Composition (%)")
        bx.set_title("Portfolio at Maximum Sharp Ratio")
        self.canvas2.draw()

        self.상태진행창.append("포트폴리오 최적화 완료!")


        # 구매 가능 수량 및 잔액 보여주기
        df_final_price = price_all.tail(1)      # 현재가(종가)만 가져오기
        money = self.text_투자금.text()          # 편리하게 사용하기 위해 새 변수에 지정


        #현재가 리스트 만들기
        last_price_list = []
        for i in range(len(self.stock_list)):
            last_price = df_final_price.iloc[0][i]
            last_price_list.append(last_price)

        # 종목별 구매 가능 수량 넣을 리스트 만들기
        stock_number = []
        for j,i in enumerate(self.stock_list):
            allot = float(money)*stock_weight_dict[i]       # 종목별 할당된 금액
            buy_number= allot//float(last_price_list[j])                              # 구매 가능 수량
            buy_number = int(buy_number)

            stock_number.append(buy_number)                      # 리스트에 넣기

        # 정보창에 주식별 예상 구매 수량과 잔액 표시하기
        self.정보창.append(" ")
        self.정보창.append("해당 포트폴리오로 구성할 시, 주식별 예상 구매 수량과 잔액은 다음과 같습니다.")
        for i in range(len(self.stock_list)):
            self.정보창.append(str(self.stock_list[i])+" : "+str(stock_number[i])+"주")

        # 구매 후 잔고 구하기
        for i in range(len(self.stock_list)):
            buy_number = stock_number[i]                         # 구매 가능 수량
            money = int(money)-int(last_price_list[i])*int(buy_number)            # 잔고

        # 잔고 및 자동 매수 문구 만들기
        self.정보창.append("잔액 : "+str(money)+" 원")
        self.정보창.append("해당 포트폴리오로 자동 매수를 원하신다면 왼쪽에 있는 '자동 매수'버튼을 눌러주세요.")
        self.정보창.append('')

        # 다른 메서드에서 쓰일 변수 전역변수 처리 : for문 내에서 지정하면 오류 발생으로 새 변수 할당 및 전역변수 처리
        global stock_number_for_buy
        stock_number_for_buy = stock_number
        global last_price_list_for_buy
        last_price_list_for_buy = last_price_list

    # 자동 매수 만들기
    def btn_buy_CLICKED(self):
        for i in range(len(self.stock_list)):
            사용자구분명 = "신규매수"
            화면번호 = '1001'
            계좌번호 = self.combo_계좌.currentText()
            주문유형int = 1                                 #신규매수
            종목코드 = str(self.stock_list[i])
            주문수량int = int(stock_number_for_buy[i])
            주문가격int = int(last_price_list_for_buy[i])
            거래구분 = '00'                                 #지정가
            원주문번호 = ''

            주문정보 = [사용자구분명, 화면번호, 계좌번호, 주문유형int, 종목코드, 주문수량int, 주문가격int, 거래구분, 원주문번호]
            self.kiwoom.dynamicCall("SendOrder(str, str, str, int, str, int, int, str, str)", 주문정보)

            self.정보창.append("["+str(self.stock_list[i])+" "+str(stock_number_for_buy[i])+"주] "+
                                "매수 주문이 접수되었습니다.")
            time.sleep(0.5)                     # 빠른 주문으로 인한 누락 방지



    def set_combo_계좌(self, 계좌리스트):
        self.combo_계좌.addItems(계좌리스트)


    def btn_login_CLICKED(self):
        self.kiwoom.dynamicCall("CommConnect()")


    def btn_login2_CLICKED(self):
        self.kiwoom.dynamicCall("KOA_Functions(s,s)", "ShowAccountWindow","")

    # 로그인 이벤트 연결
    def _event_connect(self):
        self.상태진행창.append("로그인 성공")
        account_cnt = self.kiwoom.dynamicCall("GetLoginInfo(QString)", "ACCOUNT_CNT")
        account_num = self.kiwoom.dynamicCall("GetLoginInfo(QString)", "ACCNO")
        account_id = self.kiwoom.dynamicCall("GetLoginInfo(QString)", "USER_ID")
        account_name = self.kiwoom.dynamicCall("GetLoginInfo(QString)", "USER_NAME")

        self.정보창.append("당신 id: " + account_id + ", 당신 이름: " + account_name)
        self.정보창.append("당신 계좌수: " + account_cnt)

        accountList = account_num.split(";")
        for n, x in enumerate(accountList):
            if n != int(account_cnt):
                self.정보창.append("계좌번호 {}: ".format(n + 1) + x)

        self.set_combo_계좌(accountList)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = StockWindow_Widget()
    w.show()
    app.exec_()