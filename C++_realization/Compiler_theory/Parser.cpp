#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <string.h>
#include<iostream>

// Token types
typedef enum {
    HASH,
    BEGIN, IF, THEN, WHILE, DO, END,
    etc1, etc2, etc3,
    ID, NUM,
    ADD_OP, SUB_OP, MUL_OP, DIV_OP,
    COLON, ASSIGN_OP,
    LT_OP, NE_OP, LE_OP, GT_OP, GE_OP, EQ_OP,
    SEMICOLON, LPAREN, RPAREN,
    ELSE, ERROR

} TokenType;

// 对象结构体
typedef struct {
    TokenType type;
    char* value;
} Token;

// 处理函数
void getToken();
void program();
void statementList();
void statement();
void assignmentStatement();
void conditionalStatement();
void loopStatement();
void condition();
void expression();
void term();
void factor();

// 全局变量
char input[100]; // 输入字符
int index = 0; // 字符串下标
Token currentToken;

// 取下一个标签
void getToken() {
    char currentChar = input[index++];

    // 跳过空格
    while (isspace(currentChar)) {
        currentChar = input[index++];
    }

    if (isalpha(currentChar)) {
        //ID or KEYWORD
        char buffer[100];
        int bufferIndex = 0;
        buffer[bufferIndex++] = currentChar;

        while (isalnum(input[index])) {
            buffer[bufferIndex++] = input[index++];
        }
        buffer[bufferIndex] = '\0';

        if (strcmp(buffer, "begin") == 0) {
            currentToken.type = BEGIN;
            std::cout << "识别 begin  ";
        }
        else if (strcmp(buffer, "end") == 0) {
            currentToken.type = END;
            std::cout << "识别 end    ";
        }
        else if (strcmp(buffer, "if") == 0) {
            currentToken.type = IF;
            std::cout << "识别 if ";
        }
        else if (strcmp(buffer, "then") == 0) {
            currentToken.type = THEN;
            std::cout << "识别 then   ";
        }
        else if (strcmp(buffer, "while") == 0) {
            currentToken.type = WHILE;
            std::cout << "识别 while  ";
        }
        else if (strcmp(buffer, "do") == 0) {
            currentToken.type = DO;
            std::cout << "识别 do ";
        }
        else {
            currentToken.type = ID;
            currentToken.value = _strdup(buffer);
            std::cout << "识别 ID: " <<currentToken.value<<"  ";
        }
    }
    else if (isdigit(currentChar)) {
        // NUM
        char buffer[100];
        int bufferIndex = 0;
        buffer[bufferIndex++] = currentChar;

        while (isdigit(input[index])) {
            buffer[bufferIndex++] = input[index++];
        }
        buffer[bufferIndex] = '\0';

        currentToken.type = NUM;
        currentToken.value = _strdup(buffer);
        std::cout << "识别 num: " <<currentToken.value << "  ";
    }
    else {
        // Operator or delimiter
        switch (currentChar) {
        case ':':
            if (input[index] == '=') {
                currentToken.type = ASSIGN_OP;
                index++;
                std::cout << "识别字符 " << ":=" << "  ";
            }
            else {
                currentToken.type = ERROR;
                std::cout << "ERROR1" << std::endl;
            }
            break;
        case '+':
            currentToken.type = ADD_OP;
            std::cout << "识别字符" << "+" << "  ";
            break;
        case '-':
            currentToken.type = SUB_OP;
            std::cout << "识别字符 " << "-" << "  ";
            break;
        case '*':
            currentToken.type = MUL_OP;
            std::cout << "识别字符 " << "*" << "  ";
            break;
        case '/':
            currentToken.type = DIV_OP;
            std::cout << "识别字符 " << "/" << "  ";
            break;
        case '<':
            if (input[index] == '=') {
                currentToken.type = LE_OP;
                std::cout << "识别字符 " << "<=" << "  ";
                index++;
            }
            else if (input[index] == '>') {
                currentToken.type = NE_OP;
                std::cout << "识别字符 " << "<>" << "  ";
                index++;
            }
            else {
                currentToken.type = LT_OP;
                std::cout << "识别字符 " << "<" << "  ";
            }
            break;
        case '>':
            if (input[index] == '=') {
                currentToken.type = GE_OP;
                std::cout << "识别字符 " << ">=" << "  ";
                index++;
            }
            else {
                currentToken.type = GT_OP;
                std::cout << "识别字符 " << ">" << "  ";
            }
            break;
        case '=':
            currentToken.type = EQ_OP;
            std::cout << "识别字符 " << "=" << "  ";
            break;
        case ';':
            currentToken.type = SEMICOLON;
            std::cout << "识别字符 " << ";" << "  ";
            break;
        case '(':
            currentToken.type = LPAREN;
            std::cout << "识别字符 " << "(" << "  ";
            break;
        case ')':
            currentToken.type = RPAREN;
            std::cout << "识别字符 " << ")" << "  ";
            break;
        case '#':
            currentToken.type = HASH;
            std::cout << "识别字符 " << "#" << "  ";
            break;
        default:
            currentToken.type = ERROR;
            std::cout << "ERROR， 非法字符" <<std::endl;
            break;
        }
    }
}

// 错误处理
void reportError(const char* message) {
    printf("错误语法: %s\n", message);
    exit(1);
}

// 匹配函数
void match(TokenType expectedToken) {
    if (currentToken.type == expectedToken) {
        std::cout << currentToken.type << " match sucess!" << std::endl;
        if (currentToken.type != 0) {//识别到结束符#无需在读字符串下一位
            getToken();
        }        
    }
    else {
        std::cout << "期望类别 " << expectedToken << " 实际类别 " << currentToken.type << std::endl;
        reportError("标签匹配错误");
    }
}

// 分析函数
void program() {
    match(BEGIN);
    statementList();
    match(END);
    match(HASH);
}

// 语句串函数
void statementList() {
    statement();
    while (currentToken.type != END) {
        match(SEMICOLON);
        statement();
    }
}

// 语句函数
void statement() {
    switch (currentToken.type) {
    case ID:
        assignmentStatement();
        break;
    case IF:
        conditionalStatement();
        break;
    case WHILE:
        loopStatement();
        break;
    default:
        reportError("非法语句");
        break;
    }
}

// 赋值语句函数
void assignmentStatement() {
    match(ID);
    match(ASSIGN_OP);
    expression();
}

// 条件语句函数
void conditionalStatement() {
    match(IF);
    condition();
    match(THEN);
    statement();
    match(END);
}

// 循环语句函数
void loopStatement() {
    match(WHILE);
    condition();
    match(DO);
    statement();
    match(END);
}

// 解析条件函数
void condition() {
    expression();
    switch (currentToken.type) {
    case LT_OP:
    case LE_OP:
    case NE_OP:
    case GT_OP:
    case GE_OP:
    case EQ_OP:
        match(currentToken.type);
        break;
    default:
        reportError("解析条件非法字符");
        break;
    }
    expression();
}

// 表达式函数
void expression() {
    term();
    while (currentToken.type == ADD_OP || currentToken.type == SUB_OP) {
        match(currentToken.type);
        term();
    }
}

// 项函数
void term() {
    factor();
    while (currentToken.type == MUL_OP || currentToken.type == DIV_OP) {
        match(currentToken.type);
        factor();
    }
}

// 因子函数
void factor() {
    switch (currentToken.type) {
    case ID:
        match(ID);
        break;
    case NUM:
        match(NUM);
        break;
    case LPAREN:
        match(LPAREN);
        expression();
        match(RPAREN);
        break;
    default:
        reportError("非法因子");
        break;
    }
}

int main() {
    // 输入
    // 手动输入字符串
    printf("请输入一个程序，以'#'结束:\n");
    unsigned int size = sizeof(input);
    scanf_s("%[^#]s", input, size);
    //getchar(); // 捕获换行符
    strcat_s(input, "#");
    //std::cout << input << std::endl;
    // 得到首个标签
    getToken();

    // 开始语法分析
    program();

    printf("语法分析成功！\n");

    return 0;
}
