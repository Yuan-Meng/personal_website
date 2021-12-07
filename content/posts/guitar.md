---
title: A Minimalist Setup for Zoom Guitar Lessons 
date: 2021-12-05
tags:
    - music
    - guitar
categories:
- notes
keywords:
    - music, guitar
---

# Goals + Overview (macOS)

I've been taking guitar lessons over Zoom since the pandemic began. For an electric player, it was initially challenging how I could split audio inputs/outputs multiple ways: 

1. **Input #1**: Microphone -- so I can talk to my teacher
2. **Input #2**: Guitar -- no need to explain 🎸🤟 :) 
3. **Output #1**: Headphone -- so I hear guitar effects created by my amp simulator 
3. **Output #2**: Zoom -- so my teacher hears the same effects, not just dry signals  

After some research, I found a setup that feels simple and intuitive to me. Below are a few things you need (OFC, you don't have to get the exact same models): 
1. **An audio interface**: [Focusrite Scarlett Solo USB Audio Interface](https://www.amazon.com/Focusrite-Scarlett-Audio-Interface-Tools/dp/B07QR6Z1JB?ref_=ast_sto_dp&th=1)
2. **An XLR microphone** (not USB): [Blue Baby Bottle](https://www.amazon.com/Blue-Microphones-Large-Diaphragm-Condenser-Microphone/dp/B01N7TTXZ5/ref=sr_1_1?keywords=xlr+microphone+baby+bottle&qid=1638851698&s=musical-instruments&sr=1-1)
3. **An audio-routing software**: [Loopback](https://rogueamoeba.com/loopback/) (only works for Mac)
4. **An amp simulator**: [Tonebridge](https://apps.apple.com/us/app/tonebridge-guitar-effects/id1263858588?mt=12) (more popular choices: [Logic Pro](https://apps.apple.com/us/app/logic-pro-x/id634148309?ign-itscg=20200&ign-itsct=rv_LPX_google&mt=12&mttnagencyid=b2r&mttncc=US&mttnpid=Google%20AdWords&mttnsiteid=141192&mttnsubad=lpx&mttnsubkw=ag-67301573983-ad-521528443195), [BIAS AMP](https://www.positivegrid.com/bias-amp/))
3. **Accessories**: [an XLR cable](https://www.amazon.com/AmazonBasics-Male-Female-Microphone-Cable/dp/B01JNLTTKS/ref=sr_1_2?keywords=XLR+cable&qid=1638851976&s=musical-instruments&sr=1-2), [a microphone stand](https://www.amazon.com/gp/product/B07JHCL3KS/ref=ppx_yo_dt_b_search_asin_title?ie=UTF8&psc=1), [a headphone](https://www.amazon.com/gp/product/B076BXN5MD/ref=ppx_yo_dt_b_search_asin_title?ie=UTF8&psc=1)

Now we got the gear, let's connect everything first:

1. Plug guitar + microphone + headphone into Scarlett Solo
2. Connect Scarlett Solo to your laptop via a USB
3. Register and setup your Scarlett Solo (see [instructions](https://customer.focusrite.com/en/my-hardware))  


# Loopback

Loopback allows you to stack guitar effects on top of audio interface inputs. Rename your setting for future reference. Here I chose "Tonebridge".


{{< figure src="https://www.dropbox.com/s/hvifkxhq4vhtkgq/tonebridge.png?raw=1" width="500" >}}

- **Inputs**: Choose Scarlett Solo for the microphone + Tonebridge for the guitar
- **Output**: Map both inputs to one output -- make sure the sides (L + R) align


# System Preferences

Go to "System Preferences/Sound" to set up your computer's input/output. 

{{< figure src="https://www.dropbox.com/s/k3bla9bz4t73kpa/sys_pref.png?raw=1" width="500" >}}

- **Input**: Name of your Loopback setup (in my case, "Tonebridge")
- **Output**: Scarlett Solo USB

# Tonebridge

If you like configuring guitar effects yourself, Logic Pro or BIAX AMP would be the top choice. If you just wanna replicate the exact settings of your favorite songs, I strongly recommend Tonebridge, where you can search effects by song title.

{{< figure src="https://www.dropbox.com/s/edn44tihxcohuqj/amp.png?raw=1" width="500" >}}


- **Input**: Scarlett Solo USB (Input 2)
- **Outputs**: Scarlett Solo USB (Output 1 + Output 2)

# Zoom

Head over to Zoom to finish it up. Once done, you're ready for your lessons!

{{< figure src="https://www.dropbox.com/s/1rbi5cghqf0g34r/zoom.png?raw=1" width="500" >}}

- **Speaker**: Scarlett Solo USB
- **Microphone**: Name of your Loopback setup ("Tonebridge")


