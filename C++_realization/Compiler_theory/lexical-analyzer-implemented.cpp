#include <iostream>
#include<stdio.h>
#include<string.h>
#include<stdlib.h>
#include<stack>
#include <fstream>

using namespace std;

//关键字 
string key[24] = { "and","begin","const","div","do","else","end","function","if","integer",
"not","or","procedure","program","read","real","then","type","var","while","write","for","until","repeat"};

//关键字的种别码
int keyNum[24] = {0};

//运算符和界符 
string symbol[25] = { ",",";",":",".","(",")","[","]","'","++",
"--","+","-","*","/","=","<",">","<>","<=",">=",":=","{","}","#"};

//运算符和界符的种别码 
int symbolNum[25] = {0};

//种别码初始化函数   参数分别为起点数值，下标起点，下标终点，操作数组
void initialnum(int begin, int low, int high, int* a) {
    for (int i = low; i <= high;i++) {
        a[i] = begin++;
    }
}

//判断运算符和界符 
int isSymbol(string s) {
    int i;
    for (i = 0; i < 25; i++)
        if (s == symbol[i])
            return symbolNum[i];
    return -1;
}

//判断是否为数字 
bool isNum(string s) {
    if (s >= "0" && s <= "9")
        return true;
    return false;
}

//判断是否为字母 
bool isLetter(string s)
{
    if (s >= "a" && s <= "z"||s >="A"&&s<="Z")
        return true;
    return false;
}

//判断是否为关键字,是返回种别码 
int isKeyWord(string s) {
    int i;
    for (i = 0; i < 24; i++)
        if (s == key[i])
            return keyNum[i];
    return -1;
}

// 定义一个函数，用于将栈中的字符转换为字符串
string stackToString(stack<char>& s) {
    string result = "";
    while (!s.empty()) {
        result = s.top() + result; // 逆序拼接字符
        s.pop(); // 弹出栈顶元素
    }
    return result;
}

//输出格式函数
void print(string s, int n) {
    cout << "( " << n << " , " << s << " )" << endl;
}

// 函数用于滑动窗口并移除匹配的段(若存在连续的界符号扎堆)
void slideWindow(string& a, int N) {
    int len = a.length();
    for (int windowSize = N; windowSize >= 1; windowSize--) {
        for (int i = 0; i <= len - windowSize; i++) {
            // 检查段是否与符号匹配
            int symbolIndex = isSymbol(a.substr(i, windowSize));
            if (symbolIndex != -1) {
                print(a.substr(i, windowSize), symbolIndex);
                // 从字符串中移除匹配的段
                a.erase(i, windowSize);
                len -= windowSize;
                i--; // 由于移动了字符，需要重新设置 i
            }
        }
    }
}



int main() {
    char result[20][2];
    // 打开一个文件流用于写入
    ofstream outfile("output.txt");
    if (!outfile.is_open()) {
        cerr << "无法打开文件以进行写入。" << endl;
        return 1;
    }

    // 将标准输出重定向到文件流
    streambuf* coutbuf = cout.rdbuf();
    cout.rdbuf(outfile.rdbuf());


    // 定义一个栈
    stack<char> numStack; //存储数字字符
    stack<char> wordStack; //来存储字母字符
    stack<char> symbolStack; //来存储符号字符

    //初始化种别码
    initialnum(0, 0, 20, keyNum);
    initialnum(48, 21, 23, keyNum);
    initialnum(23, 0, 24, symbolNum);

    //文件处理
    FILE* file;
    char ch, prev_ch = '\0';
    int in_comment = 0;

    // 打开文件
    if (fopen_s(&file, "test03.txt", "r") != 0) {
        printf("无法打开文件或文件不存在。\n");
        return 1;
    }

    // 逐字符读取文件内容
    while ((ch = fgetc(file)) != EOF) {
        // 如果在注释中，则跳过当前字符
        if (in_comment) {
            // 检查是否结束多行注释
            if (prev_ch == '*' && ch == '}')
                in_comment = 0;  // 注释结束
            // 未结束读取下一个
            continue;
        }
        else {
            // 检查是否进入多行注释
            if (ch == '*' && prev_ch == '{') {
                in_comment = 1;  // 进入多行注释
            }
            // 忽略单行注释//以换行符结束、{以}结束
            if (ch == '/' && prev_ch == '/') {
                while ((ch = fgetc(file)) != '\n')
                    if (ch == EOF)
                        break;
            }
            if (ch == '{') {
                while ((ch = fgetc(file)) != '}')
                    if (ch == EOF)
                        break;
            }
            else if (1) {

                //完成词法分析器主体部分
                string current_word(1, ch);  // 将当前字符转换为字符串
                // 判断当前字符类型
                
                if (isNum(current_word)) {
                    numStack.push(ch); // 将数字字符压入栈中
                }
                else
                    if (!numStack.empty()) { // 如果栈中有数字字符
                        string numStr = stackToString(numStack); // 将栈中的数字字符转换为字符串
                        print(numStr, 22);
                    }
                
                if (isLetter(current_word)) {
                    wordStack.push(ch);// 如果是字母加入栈
                }
                else { // 如果当前字符不是字母字符
                    if (!wordStack.empty()) { // 如果栈中有字母字符
                        string word = stackToString(wordStack); // 将栈中的字母字符转换为字符串
                        int keyword_num = isKeyWord(word);//判断是否为关键字
                        if (keyword_num != -1) {
                            print(word, keyword_num);
                        }
                        else print(word, 21);
                    }
                }
                
                if (!isLetter(current_word) && !isNum(current_word)&&ch!='\n'&&ch!=' ') {
                    symbolStack.push(ch);
                }
                else {
                    if (!symbolStack.empty()) { // 如果栈中有字符
                        string symbol = stackToString(symbolStack); // 将栈中的字符转换为字符串
                        slideWindow(symbol, 2);
                        //阶符或运算符处理
                        int symbol_num = isSymbol(symbol);
                        if (symbol_num != -1) {
                            print(symbol, symbol_num);
                        }
                    }
                }
            }
        }

        prev_ch = ch;
    }

    // 恢复标准输出
    cout.rdbuf(coutbuf);

    // 关闭文件
    fclose(file);


    return 0;
}
