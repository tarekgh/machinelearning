﻿// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Net.Http;
using System.Threading.Tasks;

namespace Microsoft.ML.Tokenizers.Tests
{
    public class Utils
    {
        public static async Task DownloadFileNoBackoff(string url, string fileName)
        {
            using (var client = new HttpClient() { Timeout = TimeSpan.FromMinutes(5) })
            {
                var response = await client.GetAsync(url);
                if (response.IsSuccessStatusCode)
                {
                    var stream = await response.Content.ReadAsStreamAsync();
                    var fileInfo = new FileInfo(fileName);
                    using (var fileStream = fileInfo.OpenWrite())
                    {
                        await stream.CopyToAsync(fileStream);
                    }
                }
                else
                {
                    throw new Exception("File not found");
                }
            }
        }

        public static async Task<bool> DownloadFile(string url, string fileName, int numberOfTries = 3, int delayInSeconds = 1, bool exponentialBackoff = true)
        {
            for (int i = 0; i < numberOfTries; i++)
            {
                try
                {
                    await DownloadFileNoBackoff(url, fileName);
                    return true;
                }
                catch
                {
                    await Task.Delay(TimeSpan.FromSeconds(delayInSeconds));
                    if (exponentialBackoff)
                    {
                        delayInSeconds *= 2;
                    }
                }
            }
            return false;
        }

        public static void DeleteFile(string file)
        {
            if (File.Exists(file))
            {
                try
                {
                    File.Delete(file);
                }
                catch
                {
                }
            }
        }
        public static string CreateTemporaryFile(string extension) =>
             Path.Combine(Path.GetTempPath(), Path.ChangeExtension(Guid.NewGuid().ToString(), extension));

        public static string SaveEmbeddedResourceFile(string resourceName)
        {
            string fileName = CreateTemporaryFile("txt");
            using Stream fileStream = File.Create(fileName);
            typeof(BpeTests).Assembly.GetManifestResourceStream(resourceName)!.CopyTo(fileStream);
            return fileName;
        }
    }
}
