#include <iostream>
#include <vector>
#include <string>
#include <random>
#include <chrono>

using namespace std;

const int MATR[2][2] = {{3, 0}, {5, 1}};

struct Strategy {
    virtual int move(vector<int>& my_choices, vector<int>& opp_choices) = 0;
    virtual string name() = 0;
    virtual ~Strategy() = default;
};

struct Alex : Strategy {
    int move(vector<int>& my_choices, vector<int>& opp_choices) override{
        return 0;
    };
    string name() override {
        return "Alex";
    };
};

struct Bob : Strategy {
    int move(vector<int>& my_choices, vector<int>& opp_choices) override{
        return 1;
    };
    string name() override {
        return "Bob";
    };
};

struct Clara : Strategy {
    int move(vector<int>& my_choices, vector<int>& opp_choices) override{
        if (opp_choices.empty()){
            return 0;
        } else {
            return opp_choices.back();
        }
    };
    string name() override {
        return "Clara";
    };
};

struct Denis : Strategy {
    int move(vector<int>& my_choices, vector<int>& opp_choices) override{
        if (opp_choices.empty()){
            return 0;
        } else {
            switch(opp_choices.back()){
                case 1: {
                    return 0;
                }
                case 0: {
                    return 1;
                }
                default: {
                    return 0;
                }
            }
        }
    };
    string name() override {
        return "Denis";
    };
};

struct Emma : Strategy {
    int move(vector<int>& my_choices, vector<int>& opp_choices) override{
        if (opp_choices.empty()){
            return 0;
        } else {
            if ((my_choices.size() + 1) % 20 == 0){
                return 1;
            } else {
                return 0;
            }
        }
    };
    string name() override {
        return "Emma";
    };
};

struct Frida : Strategy {
    int move(vector<int>& my_choices, vector<int>& opp_choices) override{
        if (opp_choices.empty()){
            return 0;
        } 
        for (int el : opp_choices){
            if (el == 1){
                return 1;
            }
        }
        return 0;
    };
    string name() override {
        return "Frida";
    };
};

struct Vova1 : Strategy {
    int move(vector<int>& my_choices, vector<int>& opp_choices) override {
        if (opp_choices.empty()) {
            return 0;
        }
        if (my_choices.size() == 1) {
            if (opp_choices.back() == 1) {
                return 1;
            } else {
                return 0;
            }
        }
        if (opp_choices.back() == 1 || (opp_choices.back() == 0 && my_choices[my_choices.size() - 2] == 0 && my_choices[my_choices.size() - 1] == 1)) {
            return 1;
        }
        if (opp_choices.back() == 0 || (my_choices[my_choices.size() - 2] == 1 && my_choices[my_choices.size() - 1] == 0)) {
            return 0;
        }
        return 0;
    }
    string name() override {
        return "Vova1";
    }
};

struct Vova2 : Strategy {
    int move(vector<int>& my_choices, vector<int>& opp_choices) override {
        if (opp_choices.empty()) {
            return 0;
        } 
        if (opp_choices.size() <= 4) {
            if (opp_choices.back() == 1) {
                return 1;
            }
            return 0;
        }
        int count = 0;
        int start_index = max(0, (int)opp_choices.size() - 5);
        for (int i = opp_choices.size() - 1; i >= start_index; i--) {
            if (opp_choices[i] == 1) {
                count++;
            }
        }
        if (count > 2 || opp_choices.back() == 1) {
            return 1;
        }
        return 0;
    }
    string name() override {
        return "Vova2";
    }
};

struct DrRand : Strategy {
    random_device rd;
    mt19937 gen; 
    uniform_int_distribution<int> dist;
    DrRand() : gen(rd()), dist(0, 1) {}
    
    int move(vector<int>& my_choices, vector<int>& opp_choices) override {
        return dist(gen);
    }
    string name() override {
        return "DrRand";
    }
};

struct Result {
    int scoreA;
    int scoreB;
    int maxSeries;
};

Result playGame(Strategy* a, Strategy* b) {
    Result res = {0, 0, 0};
    vector<int> MovesA;
    vector<int> MovesB;
    int series = 0;
    
    for (int i = 0; i < 200; i++) {
        int moveA = a->move(MovesA, MovesB);
        int moveB = b->move(MovesB, MovesA);
        MovesA.push_back(moveA);
        MovesB.push_back(moveB);
        int scoreA = MATR[moveA][moveB];
        int scoreB = MATR[moveB][moveA];
        res.scoreA += scoreA;
        res.scoreB += scoreB;
        if (scoreA == 5 && scoreB == 0) series++;
        else if (scoreA == 0 && scoreB == 5) series++;
        else series = 0;
        if (series > res.maxSeries) res.maxSeries = series;
    }
    return res;
}

void MyGame() {
    vector<Strategy*> StratVec = {
        new Alex(), new Bob(), new Clara(), new Denis(),
        new Emma(), new Frida(), new Vova1(), new Vova2(), new DrRand()
    };
    cout << endl << "Your Game: " << endl;
    cout << "Choose a strategy:" << endl;
    for (int i = 0; i < StratVec.size(); i++) {
        cout << i + 1 << ". " << StratVec[i]->name() << endl;
    }
    int choice;
    cin >> choice;

    Strategy* opp = StratVec[choice - 1];
    cout << "You VS " << opp->name() << endl << endl;

    vector<int> myMoves;
    vector<int> oppMoves;
    int myScore = 0;
    int oppScore = 0;
    
    for (int round = 1; round <= 20; round++) {
        cout << "Round " << round << ": ";
        int myMove;
        cin >> myMove;
        while (myMove != 0 && myMove != 1) {
            cout << "Enter 0 or 1: ";
            cin >> myMove;
        }
        int oppMove = opp->move(oppMoves, myMoves);
        myMoves.push_back(myMove);
        oppMoves.push_back(oppMove);
        myScore += MATR[myMove][oppMove];
        oppScore += MATR[oppMove][myMove];
        cout << "Score: " << myScore << " - " << oppScore << endl;
    }
    cout << "FINAL: " << myScore << " - " << oppScore << endl;
    if (myScore > oppScore){
        cout << endl << "YOU WON!!!" << endl;
    } else if (oppScore > myScore){
        cout << endl << "You lost :(" << endl;
    } else {
        cout << endl << "Draw???" << endl;
    }
    
    for (auto strat : StratVec) {
        delete strat;
    }
}

int main() {
    vector<Strategy*> StratVec = {
        new Alex(), new Bob(), new Clara(), new Denis(),
        new Emma(), new Frida(), new Vova1(), new Vova2(), new DrRand()
    };
    
    int n = StratVec.size();
    vector<vector<Result>> results(n, vector<Result>(n));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            results[i][j] = playGame(StratVec[i], StratVec[j]);
        }
    }
    vector<int> Sums;
    cout << "Result Table:" << endl << "\t";
    for (int i = 0; i < n; i++) {
        cout << StratVec[i]->name() << "\t";
    }
    cout << "\t" << "Final Sum";
    cout << endl;
    for (int i = 0; i < n; i++) {
        int sum = 0;
        cout << StratVec[i]->name() << "\t";
        for (int j = 0; j < n; j++) {
            cout << results[i][j].scoreA << "\t";
            sum += results[i][j].scoreA;
        }
        Sums.push_back(sum);
        cout << "\t" << sum;
        cout << endl;
    }
    int max = Sums[0];
    int max_i = 0;
    for (int i = 0; i < Sums.size()-1; i++){
        if (Sums[i] > max){
            max = Sums[i];
            max_i = i;
        }
    }
    cout << endl << endl;
    cout << "Winner: " << StratVec[max_i]->name() << ": " << Sums[max_i];
    cout << endl << endl;
   
    cout << endl;
    cout << "Max Series:\n\t";
    for (int i = 0; i < n; i++) cout << StratVec[i]->name() << "\t";
    cout << endl;
    for (int i = 0; i < n; i++) {
        cout << StratVec[i]->name() << "\t";
        for (int j = 0; j < n; j++) {
            cout << results[i][j].maxSeries << "\t";
        }
        cout << endl;
    }
    for (auto strat : StratVec) {
        delete strat;
    }
    cout << endl << "Wanna play game?: press (1 to start / not 1 to finish)" << endl;
    int end = 0;
    cin >> end;
    if (end == 1){
        MyGame();
    }
    return 0;
}