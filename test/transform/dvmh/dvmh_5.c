static int a[10];
int n = 10;
int n2 = 10;
void initialize() {
		for (int i = 0; i < n; i++) {
		  a[i] = 0;
		}
}

void initialize2() {
for(int j = 0; j < 10; j++) {
    for (int k = 0; k < n2; k++) {
      a[k] = k;
    }
}
}

int main(void) {
  initialize();
  initialize2();
}