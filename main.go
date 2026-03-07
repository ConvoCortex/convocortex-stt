package main

import (
	"fmt"
	"os/exec"
)

func main() {
	cmd := exec.Command("handy", "--toggle-transcription")
	err := cmd.Run()
	if err != nil {
		fmt.Println("Error:", err)
	} else {
		fmt.Println("Toggled.")
	}
}
