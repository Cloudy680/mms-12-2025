#include <iostream>
#include <random>
#include <vector>
#include <fstream>


using namespace std;

int generateSum(double p, mt19937& rand) {
    bernoulli_distribution distr(p);

    int S = 0;
    for (int i = 0; i < 100; i++){
        if (distr(rand)) {
            S++;
        }
    }
    return S;
}
double First(double p, mt19937& rand){
    double sum = 0.0;
    for (int i = 0; i < 10000; i++){
        sum += generateSum(p, rand);
    }
    sum /= 10000.0;
    return sum;
}
double Second(mt19937& rand) {
    int count = 0;
    for (int i = 0; i < 10000; i++){
        if (generateSum(0.5, rand) > 60) {
            count++;
        }
    }
    double percent = count / 10000.0;
    return percent;
}
vector<double> Third(mt19937& rand){
    vector<double> vec2;
    vector<int> vec1(10, 0);
    for (int i = 0; i < 10000; i++){
        int S = generateSum(0.5, rand);
        S /= 10;
        if (S == 10){
            vec1[9]++;
        } else {
            vec1[S]++;
        }
    }
    for (int el : vec1){
        double percent = el / 10000.0;
        vec2.push_back(percent);
    }
    return vec2;
}

pair<int, int> Fourth(double p, mt19937& rand){
    vector<int> sums;
    for (int i = 0; i < 10000; i++){
        sums.push_back(generateSum(p, rand));
    }
    sort(sums.begin(), sums.end());
    int lb = round(10000 * 0.025);
    int rb = round(10000 * 0.975);
    pair<int, int> pair(sums[lb], sums[rb]);
    return pair;
}

bool ForOne(double p, mt19937& rand){
    bernoulli_distribution distr(p);

    vector<int> vec;
    vec.reserve(100);
    for (int i = 0; i < 100; i++){
        if (distr(rand)){
            vec.push_back(1);
        } else {
            vec.push_back(0);
        }
    }
    for (int i = 0; i < vec.size(); i++){
        if (vec[i] == 1 && i <= 94){
            bool chainbroken = false;
            for (int j = i + 1; j < i + 5; j++) {
                if (vec[j] == 0){
                    chainbroken = true;
                }
            }
            if (chainbroken == false){
                return true;
            }
        }
    }
    return false;
}
double Fifth(double p, mt19937& rand){
    int count = 0;
    for (int i = 0; i < 10000; i++){
        if (ForOne(p, rand)){
            count++;
        }
    }
    double percent = count / 10000.0;
    return percent;
}
int thelongest(double p, mt19937& rand){
    bernoulli_distribution distr(p);
    int current = 0;
    int maximum = 0;
    for (int i = 0; i < 100; i++){
        if (distr(rand)){
            current++;
            if (current > maximum){
                maximum = current;
            }
        } else {
            current = 0;
        }
    }
    return maximum;
}

void Sixth(mt19937& rand) {
    ofstream file1("mean_vs_p.csv");
    ofstream file2("width_vs_p.csv");
    ofstream file3("series_prob_vs_p.csv");
    ofstream file4("max_series_vs_p.csv");
    
    file1 << "p,mean\n";
    file2 << "p,width\n";
    file3 << "p,probability\n";
    file4 << "p,max_series\n";
    
    for (double p = 0.01; p < 1.0; p += 0.01) {
        double mean = First(p, rand);
        file1 << p << "," << mean << endl;
        
        auto interval = Fourth(p, rand);
        int width = interval.second - interval.first;
        file2 << p << "," << width << endl;
        
        double series_prob = Fifth(p, rand);
        file3 << p << "," << series_prob << endl;
        
        int max_series = thelongest(p, rand);
        file4 << p << "," << max_series << endl;
        
        cout << "Progress: " << int(p * 100) << "%\r";
        cout.flush();
    }
    
    file1.close();
    file2.close();
    file3.close();
    file4.close();
    cout << endl << "Data saved tov CSV files!" << endl;
}

int main(){
    random_device rd;
    mt19937 rand(rd());
    cout << "Среднее кол-во орлов: " << First(0.5, rand) << endl << endl;
    cout << "Вероятность, чтобы орлов было 60 и больше: " << Second(rand)*100 << "%" << endl << endl;
    cout << "Вероятность выпадения числа орлов в промежутках: " << endl;
    vector<double> v = Third(rand);
    pair<int, int> pair = Fourth(0.5, rand);
    cout << "Интервал где выпало 95% всех орлов: " << "[" << pair.first << ", " << pair.second << "]" << endl << "ширина интревала: " << pair.second - pair.first << endl << endl;
    for (size_t i = 0; i < v.size(); i++){
        cout << "[" << i * 10 << ", " << (i+1)*10 << "]: " << v[i]*100 << "%" << endl;
    }
    cout << endl;
    cout << "Вероятность наличия серии из пяти орлов: " << Fifth(0.5, rand)*100 << "%" << endl << endl;
    cout << "Начинаем исследование зависимости от p..." << endl;
    Sixth(rand);
    return 0;
}