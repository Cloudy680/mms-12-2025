#include <iostream>
#include <vector>
#include <string>
#include <random>
#include <chrono>
#include <unordered_map>
#include <cmath>
#include <algorithm>

using namespace std;

const int MATR[2][2] = {{3, 0}, {5, 1}};

struct Strategy {
    virtual int move(vector<int>& my_choices, vector<int>& opp_choices) = 0;
    virtual string name() = 0;
    virtual bool IsRand() = 0;
    virtual ~Strategy() = default;
};

struct Alex : Strategy {
    int move(vector<int>& my_choices, vector<int>& opp_choices) override{
        return 0;
    };
    bool IsRand() override {
        return false;
    }
    string name() override {
        return "Alex";
    };
};

struct Bob : Strategy {
    int move(vector<int>& my_choices, vector<int>& opp_choices) override{
        return 1;
    };
    bool IsRand() override {
        return false;
    }
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
    bool IsRand() override {
        return false;
    }
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
    bool IsRand() override {
        return false;
    }
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
    bool IsRand() override {
        return false;
    }
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
    bool IsRand() override {
        return false;
    }
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
    bool IsRand() override {
        return false;
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
    bool IsRand() override {
        return false;
    }
    string name() override {
        return "Vova2";
    }
};

struct Hank : Strategy {
    random_device rd;
    mt19937 gen; 
    bernoulli_distribution dist;
    Hank() : gen(rd()), dist(0.5) {}
    int move(vector<int>& my_choices, vector<int>& opp_choices) override {
        return dist(gen);
    }
    bool IsRand() override {
        return true;
    }
    string name() override {
        return "Hank";
    }
};

struct Ivan : Strategy {
    random_device rd;
    mt19937 gen; 
    bernoulli_distribution dist;
    Ivan() : gen(rd()), dist(0.1) {}
    int move(vector<int>& my_choices, vector<int>& opp_choices) override {
        return dist(gen);
    }
    bool IsRand() override {
        return true;
    }
    string name() override {
        return "Ivan";
    }
};

struct Jack : Strategy {
    random_device rd;
    mt19937 gen; 
    bernoulli_distribution dist;
    Jack() : gen(rd()), dist(0.25) {}    
    int move(vector<int>& my_choices, vector<int>& opp_choices) override {
        if (opp_choices.empty()) {
            return 0;
        }
        if (opp_choices.back() == 0) {
            return 0;
        } else {
            return !dist(gen);
        }
    }
    bool IsRand() override {
        return true;
    }
    string name() override {
        return "Jack";
    }
};

struct Kevin : Strategy {
    random_device rd;
    mt19937 gen; 
    bernoulli_distribution dist;
    Kevin() : gen(rd()), dist(0.25) {}    
    int move(vector<int>& my_choices, vector<int>& opp_choices) override {
        if (opp_choices.empty()) {
            return 0;
        }
        if (dist(gen)) {
            return 1 - opp_choices.back();
        } else {
            return opp_choices.back();
        }
    }
    bool IsRand() override {
        return true;
    }
    string name() override {
        return "Kevin";
    }
};

struct Lucas : Strategy {
    random_device rd;
    mt19937 gen; 
    uniform_int_distribution<int> dist;
    int period;
    Lucas() : gen(rd()), dist(1, 50) {
        period = dist(gen);
    }    
    int move(vector<int>& my_choices, vector<int>& opp_choices) override {
        if (my_choices.size() % period == 0) {
            return 1;
        } else {
            return 0;
        }
    }
    bool IsRand() override {
        return true;
    }
    string name() override {
        return "Lucas";
    }
};

struct Max : Strategy {
    random_device rd;
    mt19937 gen; 
    uniform_int_distribution<int> dist;
    bernoulli_distribution move_d;
    int curr_move = 0;
    int count;
    Max() : gen(rd()), dist(0, 20) {
        count = dist(gen);
    }    
    int move(vector<int>& my_choices, vector<int>& opp_choices) override {
        if (count == 0) {
            count = dist(gen);
            curr_move = 1 - curr_move;
        }
        count--;
        return curr_move;
    }
    bool IsRand() override {
        return true;
    }
    string name() override {
        return "Max";
    }
};

struct VovaR : Strategy {
    random_device rd;
    mt19937 gen; 
    int move(vector<int>& my_choices, vector<int>& opp_choices) override {
        if (opp_choices.empty()) {
            return 0;
        }
        int betrs = 0;
        for (int move : opp_choices) {
            if (move == 1) {
                betrs++;
            }
        }
        double prob_betr = static_cast<double>(betrs) / opp_choices.size();
        double my_betr = 0.3 + 0.4 * prob_betr;
        if (!opp_choices.empty() && opp_choices.back() == 1) {
            my_betr = my_betr + 0.1;
        }
        bernoulli_distribution dist(my_betr);
        return dist(gen);
    }
    bool IsRand() override {
        return true;
    }
    string name() override {
        return "VovaR";
    }
};

struct RandResult {
    double mid;
    double med;
    double mod;
    double disp;
    double maxSeries;
};

struct Result {
    int scoreA;
    int scoreB;
    int maxSeries;
    RandResult rand_r;
};

double Middle(const vector<int>& res){
    double sum = 0.0;
    for (int el : res){
        sum += el;
    }
    sum /= res.size();
    return sum;
}

double Median(vector<int> res) {
    sort(res.begin(), res.end());
    if (res.size() % 2 == 0) {
        return (res[res.size()/2 - 1] + res[res.size()/2]) / 2.0;
    } else {
        return (res[res.size()/2]);
    }
}

int Modal(const vector<int>& res){
    unordered_map<int, int> map;
    for (int el : res){
        map[el]++;
    }
    int res_el = res[0];
    int max_c = 0;
    for (auto& [el, count] : map){
        if (count > max_c){
            max_c = count;
            res_el = el;
        }
    }
    return res_el;
}

double Dispertion(const vector<int>& res){
    double mid = Middle(res);
    double sum = 0.0;
    for (int el : res){
        sum += pow(el - mid, 2);
    }
    sum /= (res.size() - 1);
    return sum;
}

Result playNonRandGame(Strategy* a, Strategy* b) {
    Result r = {0, 0, 0};
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
        r.scoreA += scoreA;
        r.scoreB += scoreB;
        if (scoreA == 5 && scoreB == 0) {
            series++;
        }
        else if (scoreA == 0 && scoreB == 5) {
            series++;
        }
        else {
            series = 0;
        }
        if (series > r.maxSeries) {
            r.maxSeries = series;
        }
    }
    return r;
}

RandResult playRandGame(Strategy* a, Strategy* b) {
    RandResult res = {0, 0, 0, 0, 0};
    vector<int> vecA;
    vector<int> vecSeries;
    for (int i = 0; i < 1000; i++){
        Result r = playNonRandGame(a, b);
        vecA.push_back(r.scoreA);
        vecSeries.push_back(r.maxSeries);
    }
    res.mid = Middle(vecA);
    res.med = Median(vecA);
    res.mod = Modal(vecA);
    res.disp = Dispertion(vecA);
    res.maxSeries = Middle(vecSeries);
    return res;
}

Result playGame(Strategy* a, Strategy* b){
    Result result = {0, 0, 0};
    if (a->IsRand() == true || b->IsRand() == true){
        RandResult res = playRandGame(a, b);
        result.rand_r = res;
        result.scoreA = res.med;
        result.maxSeries = res.maxSeries;
    } else {
        result = playNonRandGame(a, b);
    }
    return result;
}

void MyGame() {
    vector<Strategy*> StratVec = {
        new Alex(), new Bob(), new Clara(), new Denis(),
        new Emma(), new Frida(), new Vova1(), new Vova2(), 
        new Hank(), new Ivan(), new Jack(), new Kevin(),
        new Lucas(), new Max(), new VovaR()
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

void Work(vector<Strategy*> StratVec){
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
    cout << endl;
    for (int i = 0; i < n; i++) {
        int sum = 0;
        cout << StratVec[i]->name() << "\t";
        for (int j = 0; j < n; j++) {
            cout << results[i][j].scoreA << "\t";
            sum += results[i][j].scoreA;
        }
        Sums.push_back(sum);
        cout << endl;
    }
    cout << endl << "Sums: " << endl;
    for (auto el : StratVec){
        cout << "\t" << el->name();
    }
    cout << endl;
    for (int el : Sums){
        cout << "\t" << el;
    }
    int max = Sums[0];
    int max_i = 0;
    for (int i = 0; i < Sums.size(); i++){
        if (Sums[i] > max){
            max = Sums[i];
            max_i = i;
        }
    }
    cout << endl << endl;
    cout << "Winner: " << StratVec[max_i]->name() << ": " << Sums[max_i];
    cout << endl << endl;

    cout << endl << "Statistics for random:" << endl;
    for (int i = 0; i < n; i++) {
        if (StratVec[i]->IsRand() == true) {
            cout << StratVec[i]->name() << " statistics:" << endl;
            for (int j = 0; j < n; j++) {
                if (StratVec[i]->IsRand() == true || StratVec[j]->IsRand() == true) {
                    cout << "  vs " << StratVec[j]->name() << ": ";
                    cout << "mid=" << results[i][j].rand_r.mid << ", ";
                    cout << "med=" << results[i][j].rand_r.med << ", ";
                    cout << "mod=" << results[i][j].rand_r.mod << ", ";
                    cout << "disp=" << results[i][j].rand_r.disp << endl << endl;
                }
            }
            cout << endl;
        }
    }
   
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
    
    cout << endl << "Wanna play game?: press (1 to start / not 1 to finish)" << endl;
    int end = 0;
    cin >> end;
    if (end == 1){
        MyGame();
    }
    
    for (auto strat : StratVec) {
        delete strat;
    }
}

int main() {
    vector<Strategy*> StratVec = {
        new Alex(), new Bob(), new Clara(), new Denis(),
        new Emma(), new Frida(), new Vova1(), new Vova2(), 
        new Hank(), new Ivan(), new Jack(), new Kevin(),
        new Lucas(), new Max(), new VovaR()
    };
    Work(StratVec);
}